"""Independent wire fixtures and hand-computed books (ns, shares, price units)."""

from collections import Counter
from dataclasses import fields
from decimal import Decimal
import gzip
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lob_latency.events import DUMMY_PRICE, merge_events
from lob_latency.io_lobster import load_lobster
from lob_latency.itch import RECORD_DTYPE, SESSION_START_NS, load_records, reconstruct

ROOT = Path(__file__).resolve().parents[1]


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def filter_binary(tmp_path_factory):
    binary = tmp_path_factory.mktemp("itch_build") / "itch_filter"
    subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(ROOT / "tools/itch_filter.c"), "-o", str(binary)], check=True)
    return binary


def wire(kind, ts, locate=1, *, ref=0, side='B', shares=0, price=0,
         symbol="AAPL", new_ref=0, printable='Y'):
    """Pack full official ITCH messages independently of the production dtype."""
    payload = kind.encode() + struct.pack('>HH', locate, 0) + ts.to_bytes(6, 'big')
    stock = symbol.encode().ljust(8)
    if kind == 'R':
        payload += (stock + b'QN' + struct.pack('>I', 100) + b'NC  PNN1N'
                    + struct.pack('>I', 1) + b'N')
    elif kind in 'AFP':
        payload += struct.pack('>QcI8sI', ref, side.encode(), shares, stock, price)
        payload += b'MPID' if kind == 'F' else struct.pack('>Q', 42) if kind == 'P' else b''
    elif kind in 'ECX':
        payload += struct.pack('>QI', ref, shares)
        if kind in 'EC':
            payload += struct.pack('>Q', 42)
        if kind == 'C':
            payload += struct.pack('>cI', printable.encode(), price)
    elif kind == 'D':
        payload += struct.pack('>Q', ref)
    elif kind == 'U':
        payload += struct.pack('>QQII', ref, new_ref, shares, price)
    elif kind == 'Q':
        payload += struct.pack('>Q8sIQc', shares, stock, price, 42, b'O')
    elif kind == 'H':
        payload += stock + b'H' + b' ' + b'T1  '
    elif kind == 'S':
        payload += b'O'
    else:
        raise ValueError(kind)
    expected = dict(R=39, A=36, F=40, P=44, E=31, C=36, X=23, D=19, U=35, Q=40, H=25, S=12)
    assert len(payload) == expected[kind]
    return struct.pack('>H', len(payload)) + payload


def run_filter(binary, tmp_path, raw, symbols="AAPL"):
    output = tmp_path / "filtered"
    subprocess.run([str(binary), str(output), symbols], input=raw, check=True, capture_output=True)
    return output, json.loads((output / "filter_counts.json").read_text())


def synthetic_stream():
    t = SESSION_START_NS
    messages = [
        wire('R', 0), wire('R', 0, locate=2, symbol="MSFT"),
        wire('A', t - 10, ref=1, shares=100, price=10_000),
        wire('F', t - 9, ref=2, side='S', shares=80, price=10_200),
        wire('A', t, ref=3, shares=50, price=9_900),
        wire('A', t + 1, ref=4, side='S', shares=20, price=10_300),
        wire('E', t + 2, ref=1, shares=30),
        wire('C', t + 3, ref=2, shares=80, price=10_150, printable='N'),
        wire('U', t + 4, ref=1, new_ref=5, shares=40, price=10_100),
        wire('X', t + 5, ref=5, shares=10),
        wire('X', t + 6, ref=3, shares=50),  # Full cancel remains LOBSTER type 2.
        wire('D', t + 7, ref=4),
        wire('P', t + 8, shares=7, price=10_050),
        wire('Q', t + 9, shares=123, price=10_100),
        wire('H', t + 10),
        wire('A', t + 11, locate=2, ref=11, symbol="MSFT", shares=5, price=20_000),
        wire('S', t + 12, locate=0),
    ]
    return b''.join(messages) + wire('A', t + 13, ref=99, shares=9, price=10_000)[:-4]


def test_filter_wire_and_hand_calculated_replay(filter_binary, tmp_path):
    raw = synthetic_stream()
    output, counts = run_filter(filter_binary, tmp_path, raw)
    records = load_records(output / "AAPL.bin")
    assert RECORD_DTYPE.itemsize == 40
    assert [RECORD_DTYPE.fields[k][1] for k in RECORD_DTYPE.names] == [0, 8, 16, 24, 28, 32, 33, 34, 35]
    assert counts["bytes_read"] == len(raw)
    assert counts["total_messages"] == 17
    assert counts["input_ended_mid_message"]
    assert counts["counts_by_type"] == dict(R=2, A=4, F=1, E=1, C=1, U=1, X=2, D=1, P=1, Q=1, H=1, S=1)
    assert counts["symbols"]["AAPL"]["records"] == 13
    assert set(counts["symbols"]) == {"AAPL"}
    assert not (output / "MSFT.bin").exists()
    assert bytes(records[0]) == struct.pack('<QQQIIBBB5x', SESSION_START_NS - 10, 1, 0, 100, 10_000, ord('A'), ord('B'), 0)
    c = records[5]
    assert (c['type'], c['printable'], c['price']) == (ord('C'), ord('N'), 10_150)
    result = reconstruct(output / "AAPL.bin", output, "AAPL", batch_rows=2)
    messages = pl.read_parquet(output / "AAPL_message_10.parquet")
    book = pl.read_parquet(output / "AAPL_orderbook_10.parquet").to_numpy()
    expected_rows = [
        (1, 3, 50, 9_900, 1), (1, 4, 20, 10_300, -1),
        (4, 1, 30, 10_000, 1), (4, 2, 80, 10_150, -1),
        (3, 1, 70, 10_000, 1), (1, 5, 40, 10_100, 1),
        (2, 5, 10, 10_100, 1), (2, 3, 50, 9_900, 1),
        (3, 4, 20, 10_300, -1), (5, 0, 7, 10_050, 0),
        (6, 0, 123, 10_100, 0), (7, 0, 0, 0, 0),
    ]
    assert messages.select(messages.columns[1:]).rows() == expected_rows
    assert messages['time'].to_list()[:2] == [Decimal('34200.000000000'), Decimal('34200.000000001')]
    empty = [DUMMY_PRICE, 0, -DUMMY_PRICE, 0]
    expected_top_two = [
        [10_200, 80, 10_000, 100, DUMMY_PRICE, 0, 9_900, 50],
        [10_200, 80, 10_000, 100, 10_300, 20, 9_900, 50],
        [10_200, 80, 10_000, 70, 10_300, 20, 9_900, 50],
        [10_300, 20, 10_000, 70, DUMMY_PRICE, 0, 9_900, 50],
        [10_300, 20, 9_900, 50, *empty],
        [10_300, 20, 10_100, 40, DUMMY_PRICE, 0, 9_900, 50],
        [10_300, 20, 10_100, 30, DUMMY_PRICE, 0, 9_900, 50],
        [10_300, 20, 10_100, 30, *empty],
        [DUMMY_PRICE, 0, 10_100, 30, *empty],
    ]
    expected_top_two += [expected_top_two[-1]] * 3
    np.testing.assert_array_equal(book, [row + empty * 8 for row in expected_top_two])
    assert result['emitted_rows'] == 12
    assert result['final_open_order_count'] == 1
    assert result['unknown_order_refs'] == result['negative_shares'] == 0
    assert result['crossed_or_locked']['count'] == 0
    loaded = load_lobster(output / "AAPL_message_10.parquet", output / "AAPL_orderbook_10.parquet")
    assert loaded.raw_rows == 12 and loaded.dropped_halts == 1
    assert len(merge_events(loaded).time_ns) == 10  # Replace rows share a timestamp.


@pytest.mark.parametrize("tail,expected", [(b'', False), (b'\0', True), (b'\0\x24A\0', True)])
def test_clean_and_truncated_eof(filter_binary, tmp_path, tail, expected):
    output, counts = run_filter(filter_binary, tmp_path, wire('R', 0) + tail)
    assert counts['input_ended_mid_message'] is expected
    assert len(load_records(output / 'AAPL.bin')) == 0
    result = reconstruct(output / 'AAPL.bin', output, 'AAPL')
    assert result['emitted_rows'] == 0
    data = load_lobster(output / 'AAPL_message_10.parquet', output / 'AAPL_orderbook_10.parquet')
    assert data.book.shape == (0, 40) and data.raw_rows == 0


def test_full_execution_and_session_edges(filter_binary, tmp_path):
    close = 57_600_000_000_000
    raw = wire('R', 0) + wire('A', SESSION_START_NS - 1, ref=1, shares=10, price=10_000)
    raw += wire('E', SESSION_START_NS, ref=1, shares=10)
    raw += wire('P', close, shares=5, price=10_000)
    raw += wire('A', close + 1, ref=2, shares=10, price=10_000)
    output, _ = run_filter(filter_binary, tmp_path, raw)
    result = reconstruct(output / 'AAPL.bin', output, 'AAPL')
    assert result['lobster_counts_by_type'] == {'4': 1, '5': 1}
    assert result['final_open_order_count'] == 1
    book = pl.read_parquet(output / 'AAPL_orderbook_10.parquet').to_numpy()
    np.testing.assert_array_equal(book, np.tile([DUMMY_PRICE, 0, -DUMMY_PRICE, 0], (2, 10)))


def test_depth_visibility_and_out_of_range_execution(filter_binary, tmp_path):
    t = SESSION_START_NS
    raw = wire('R', 0)
    for i in range(12):
        raw += wire('A', t - 20 + i, ref=i + 1, shares=10, price=10_000 - i * 100)
    raw += wire('X', t, ref=12, shares=1)  # Outside top ten: omit.
    raw += wire('E', t + 1, ref=12, shares=1)  # Always emit an execution.
    raw += wire('D', t + 2, ref=1)  # The eleventh bid enters the requested range.
    output, _ = run_filter(filter_binary, tmp_path, raw)
    result = reconstruct(output / 'AAPL.bin', output, 'AAPL')
    assert result['lobster_counts_by_type'] == {'3': 1, '4': 1}
    book = pl.read_parquet(output / 'AAPL_orderbook_10.parquet').to_numpy()
    np.testing.assert_array_equal(book[:, 38], [9_100, 9_000])


def test_integrity_counts_and_observed_durations(filter_binary, tmp_path):
    t = SESSION_START_NS
    raw = b''.join([
        wire('R', 0), wire('A', t - 10, ref=1, shares=10, price=10_000),
        wire('A', t - 9, ref=2, side='S', shares=10, price=10_000),  # Locked at open.
        wire('P', t + 2, shares=1, price=10_000),
        wire('U', t + 5, ref=2, new_ref=3, shares=10, price=9_900),  # Crossed.
        wire('D', t + 8, ref=3),
        wire('X', t + 9, ref=1, shares=12),  # Over-removal, clamped to ten shares.
        wire('D', t + 10, ref=999), wire('U', t + 11, ref=998, new_ref=99),
        wire('E', t + 12, ref=997, shares=1),
    ])
    output, _ = run_filter(filter_binary, tmp_path, raw)
    result = reconstruct(output / 'AAPL.bin', output, 'AAPL')
    assert result['unknown_order_refs'] == 3
    assert result['unknown_order_refs_by_type'] == {'D': 1, 'U': 1, 'E': 1}
    assert result['negative_shares'] == 1 and result['final_open_order_count'] == 0
    assert result['locked'] == dict(count=1, intervals=1, longest_duration_ns=5, total_duration_ns=5)
    assert result['crossed'] == dict(count=1, intervals=1, longest_duration_ns=3, total_duration_ns=3)
    assert result['crossed_or_locked'] == dict(count=2, intervals=1, longest_duration_ns=8, total_duration_ns=8)


def test_duration_stops_at_observation_end(filter_binary, tmp_path):
    t = SESSION_START_NS
    raw = wire('R', 0) + wire('A', t - 2, ref=1, shares=1, price=10_000)
    raw += wire('A', t - 1, ref=2, shares=1, price=10_000, side='S')
    output, _ = run_filter(filter_binary, tmp_path, raw)
    result = reconstruct(output / 'AAPL.bin', output, 'AAPL', observation_end_ns=t + 100)
    assert result['locked']['longest_duration_ns'] == 100


@pytest.mark.parametrize('case', ['zero', 'short', 'q_overflow', 'duplicate_symbols', 'invalid_symbol'])
def test_filter_rejects_invalid_input(filter_binary, tmp_path, case):
    inputs = {'zero': b'\0\0', 'short': b'\0\x01A',
              'q_overflow': wire('R', 0) + wire('Q', 1, shares=2**32),
              'duplicate_symbols': b'', 'invalid_symbol': b''}
    symbols = 'AAPL,AAPL' if case == 'duplicate_symbols' else '../bad' if case == 'invalid_symbol' else 'AAPL'
    result = subprocess.run([str(filter_binary), str(tmp_path / 'out'), symbols],
                            input=inputs[case], capture_output=True)
    assert result.returncode != 0 and result.stderr


def test_record_validation(tmp_path):
    path = tmp_path / 'bad.bin'
    path.write_bytes(b'\0')
    with pytest.raises(ValueError, match='Partial'):
        load_records(path)
    records = np.zeros(2, dtype=RECORD_DTYPE)
    records['type'] = ord('H')
    records['ts_ns'] = [10, 9]
    records.tofile(path)
    with pytest.raises(ValueError, match='chronological'):
        reconstruct(path, tmp_path, 'AAPL')


def test_csv_parquet_loader_parity(tmp_path):
    message = tmp_path / 'message.csv'
    message.write_text('35100.000000001,1,18446744073709551615,10,10100,1\n'
                       '35100.000000002,7,0,0,0,0\n'
                       '35100.000000003,4,1,5,10000,-1\n')
    book = tmp_path / 'book.csv'
    np.savetxt(book, np.tile([10_100, 10, 10_000, 20], (3, 10)), delimiter=',', fmt='%d')
    mp, bp = tmp_path / 'message.parquet', tmp_path / 'book.parquet'
    pl.read_csv(message, has_header=False,
                new_columns=['time', 'type', 'order_id', 'size', 'price', 'direction'],
                schema_overrides={'time': pl.String, 'order_id': pl.UInt64}).write_parquet(mp)
    pl.read_csv(book, has_header=False).write_parquet(bp)
    csv, parquet = load_lobster(message, book), load_lobster(mp, bp)
    for field in fields(csv):
        np.testing.assert_array_equal(getattr(csv, field.name), getattr(parquet, field.name))
    # Decimal seconds use the same exact nanoseconds as CSV strings.
    table = pq.read_table(mp).set_column(0, 'time', pa.array([
        Decimal('35100.000000001'), Decimal('35100.000000002'), Decimal('35100.000000003')],
        type=pa.decimal128(14, 9)))
    table = table.set_column(1, 'type', table.column(1).cast(pa.int8()))
    table = table.set_column(5, 'direction', table.column(5).cast(pa.int8()))
    pq.write_table(table, mp)
    parquet = load_lobster(mp, bp)
    for field in fields(csv):
        expected, actual = getattr(csv, field.name), getattr(parquet, field.name)
        np.testing.assert_array_equal(expected, actual)
        if isinstance(expected, np.ndarray):
            assert expected.dtype == actual.dtype
    table = table.set_column(0, 'time', table.column(0).cast(pa.float64()))
    pq.write_table(table, mp)
    with pytest.raises(ValueError, match='exact decimal or string'):
        load_lobster(mp, bp)


def test_cli_small_gzip_and_keep_bin(tmp_path):
    source = tmp_path / 'S120825-v50.txt.gz'
    source.write_bytes(gzip.compress(synthetic_stream(), mtime=0))
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/itch_to_lobster.py'),
               '--input', str(source), '--symbols', 'AAPL,MSFT', '--out', str(tmp_path / 'converted')]
    subprocess.run(command + ['--keep-bin'], check=True, capture_output=True, text=True)
    output = tmp_path / 'converted/2025-12-08'
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['weekday'] == 'Monday'
    assert not manifest['complete_input']  # Complete gzip, partial final ITCH frame.
    assert not manifest['filter']['gzip_truncated']
    assert (output / 'AAPL.bin').exists() and (output / 'MSFT.bin').exists()
    assert manifest['integrity']['AAPL']['emitted_rows'] == 12
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert not (output / 'AAPL.bin').exists()
    subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/symbol_stats.py'),
                    '--input', str(output), '--out', str(tmp_path / 'stats')],
                   check=True, capture_output=True, text=True)
    stats = pl.read_csv(tmp_path / 'stats/symbol_stats.csv')
    assert stats.height == 2 and stats['merged_events'].sum() == 0


def test_cli_accepts_truncated_gzip_but_rejects_corruption(filter_binary, tmp_path):
    module = script_module('itch_to_lobster')
    source = tmp_path / 'S120825_head64MB.gz'
    compressed = gzip.compress(wire('R', 0) + wire('H', SESSION_START_NS), mtime=0)
    source.write_bytes(compressed[:-8])  # No gzip trailer, but complete deflate messages.
    output = tmp_path / 'out'
    result = module.filter_input(filter_binary, source, output, ['AAPL'])
    assert result['gzip_truncated'] and not result['counts']['input_ended_mid_message']
    source.write_bytes(b'not a gzip')
    with pytest.raises(RuntimeError, match='gzip failed'):
        module.filter_input(filter_binary, source, output, ['AAPL'])
    damaged = bytearray(compressed)
    damaged[-8] ^= 0xff
    source.write_bytes(damaged)
    with pytest.raises(RuntimeError, match='gzip failed'):
        module.filter_input(filter_binary, source, output, ['AAPL'])


def python_reference_counts(source, symbols):
    """Pure-Python framing parser over the same system-gzip byte stream.

    System gzip and Python gzip expose different tails from incomplete deflate
    blocks on macOS. Sharing only decompression makes the framing comparison
    well-defined; no production record/layout/filter code is used here.
    """
    counts = Counter()
    by_symbol = {symbol: Counter() for symbol in symbols}
    kept = Counter()
    mapping = {}
    total_bytes = 0
    partial = False
    gzip_truncated = False
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(['gzip', '-dc', str(source)], stdout=subprocess.PIPE, stderr=errors)
        with process.stdout as stream:
            while True:
                header = stream.read(2)
                total_bytes += len(header)
                if len(header) != 2:
                    partial = bool(header)
                    break
                length = int.from_bytes(header, 'big')
                message = stream.read(length)
                total_bytes += len(message)
                if len(message) != length:
                    partial = True
                    break
                kind = chr(message[0])
                counts[kind] += 1
                locate = int.from_bytes(message[1:3], 'big')
                if kind == 'R':
                    mapping[locate] = message[11:19].decode().rstrip(' ')
                symbol = mapping.get(locate)
                if symbol in by_symbol:
                    by_symbol[symbol][kind] += 1
                    if kind in 'AFECXDUPQH':
                        kept[symbol] += 1
        code = process.wait()
        errors.seek(0)
        diagnostics = errors.read().decode()
        gzip_truncated = code != 0 and 'unexpected end of file' in diagnostics
        assert code == 0 or gzip_truncated
    return dict(counts), {s: dict(c) for s, c in by_symbol.items()}, kept, gzip_truncated, partial, total_bytes


def test_head_sample_filter_matches_independent_python(filter_binary, tmp_path):
    source = ROOT / 'data/raw/itch_head/S120825_head64MB.gz'
    if not source.exists():
        pytest.skip('Local head sample absent')
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']
    module = script_module('itch_to_lobster')
    result = module.filter_input(filter_binary, source, tmp_path / 'head', symbols)
    counts, reference, kept, truncated, partial, total_bytes = python_reference_counts(source, symbols)
    assert truncated and result['gzip_truncated']
    assert result['counts']['counts_by_type'] == counts
    assert result['counts']['bytes_read'] == total_bytes
    assert result['counts']['input_ended_mid_message'] is partial
    for symbol in symbols:
        assert result['counts']['symbols'][symbol]['counts_by_type'] == reference[symbol]
        assert result['counts']['symbols'][symbol]['records'] == kept[symbol]


def test_symbol_statistics_by_hand():
    from lob_latency.io_lobster import LobsterData
    module = script_module('symbol_stats')
    t = 35_100 * 10**9
    # Two messages at the same timestamp; final state has a one-tick spread.
    books = np.tile([10_100, 10, 10_000, 20], (5, 10))
    books[1, 0] = 10_200
    books[3:, 0] = 10_300
    data = LobsterData(t + np.array([-1, 0, 0, 2, 6]) * 10**9,
                       np.ones(5, dtype=int), np.ones(5), books, 5, 0)
    stats = module.symbol_day_statistics(data, 'AAPL', 'synthetic', observation_end_ns=t + 10 * 10**9)
    assert stats['messages'] == 4 and stats['merged_events'] == 3
    assert stats['mean_inter_event_gap_seconds'] == 3
    assert stats['median_inter_event_gap_seconds'] == 3
    assert stats['mid_changes_per_event'] == pytest.approx(1 / 3)
    assert stats['mean_spread_ticks'] == pytest.approx(7 / 3)
    assert stats['one_tick_time_share'] == pytest.approx(2 / 10)
    assert stats['mean_mid_price_dollars'] == pytest.approx((1.005 + 1.015 + 1.015) / 3)
    assert stats['tick_size_class'] == 'medium'
    assert stats['observed_book_seconds'] == 10


@pytest.mark.parametrize('spread,expected', [(1.5, 'large'), (3, 'small'), (2, 'medium')])
def test_tick_class_boundaries(spread, expected):
    from lob_latency.io_lobster import LobsterData
    book = np.tile([10_000 + int(spread * 100), 1, 10_000, 1], (1, 10))
    data = LobsterData(np.array([35_100 * 10**9]), np.array([1]), np.array([1]), book, 1, 0)
    result = script_module('symbol_stats').symbol_day_statistics(data, 'AAPL', 'synthetic')
    assert result['tick_size_class'] == expected
    assert result['one_tick_time_share'] is None  # No observation interval.
