/* C99 streaming ITCH 5.0 filter. Times: ns; prices: dollars * 10000.
 * Wire records are exactly 40 little-endian bytes, never a native C struct.
 * See RECORD_DTYPE in src/lob_latency/itch.py. */
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {
    char name[9];
    FILE *out;
    uint64_t counts[256];
    uint64_t records;
} Symbol;

static uint64_t be(const unsigned char *p, unsigned n) {
    uint64_t x = 0;
    while (n--) x = (x << 8) | *p++;
    return x;
}

static void le(unsigned char *p, uint64_t x, unsigned n) {
    while (n--) { *p++ = (unsigned char)x; x >>= 8; }
}

static void counts_json(FILE *f, const uint64_t *counts) {
    int comma = 0;
    fputc('{', f);
    for (unsigned i = 0; i < 256; ++i) {
        if (!counts[i]) continue;
        /* Escaped JSON keys also support unexpected message-type bytes. */
        fprintf(f, "%s\"\\u%04x\":%" PRIu64, comma ? "," : "", i, counts[i]);
        comma = 1;
    }
    fputc('}', f);
}

static unsigned min_length(unsigned char type) {
    switch (type) {
        case 'R': return 39;
        case 'A': return 36;
        case 'F': return 40;
        case 'E': return 31;
        case 'C': return 36;
        case 'X': return 23;
        case 'D': return 19;
        case 'U': return 35;
        case 'P': return 44;
        case 'Q': return 40;
        case 'H': return 25;
        default: return 1;
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: itch_filter <out_dir> <SYM1,SYM2,...>\n");
        return 2;
    }
    size_t capacity = strlen(argv[2]) + 1, n_symbols = 0;
    Symbol *symbols = calloc(capacity, sizeof(*symbols));
    char *names = malloc(capacity);
    if (!symbols || !names) { fprintf(stderr, "allocation failed\n"); return 1; }
    memcpy(names, argv[2], capacity);
    /* Parse without strtok so empty entries are rejected. */
    char *start = names;
    for (char *p = names;; ++p) {
        if (*p != ',' && *p != '\0') continue;
        int end = *p == '\0';
        size_t len = (size_t)(p - start);
        if (len == 0 || len > 8) { fprintf(stderr, "invalid symbol\n"); return 2; }
        for (size_t j = 0; j < len; ++j)
            if (!isalnum((unsigned char)start[j]) && !strchr("._-", start[j])) {
                fprintf(stderr, "invalid symbol\n"); return 2;
            }
        memcpy(symbols[n_symbols].name, start, len);
        for (size_t j = 0; j < n_symbols; ++j)
            if (!strcmp(symbols[j].name, symbols[n_symbols].name)) {
                fprintf(stderr, "duplicate symbol\n"); return 2;
            }
        ++n_symbols;
        if (end) break;
        start = p + 1;
    }
    if (mkdir(argv[1], 0777) && errno != EEXIST) { perror(argv[1]); return 1; }
    size_t path_size = strlen(argv[1]) + 64;
    char *path = malloc(path_size);
    if (!path) return 1;
    for (size_t i = 0; i < n_symbols; ++i) {
        snprintf(path, path_size, "%s/%s.bin", argv[1], symbols[i].name);
        symbols[i].out = fopen(path, "wb");
        if (!symbols[i].out) { perror(path); return 1; }
        setvbuf(symbols[i].out, NULL, _IOFBF, 65536);
    }
    int locates[65536];
    for (size_t i = 0; i < 65536; ++i) locates[i] = -1;
    uint64_t counts[256] = {0}, bytes = 0, total = 0;
    uint64_t first_ts = 0, last_ts = 0;
    int have_ts = 0;
    int truncated = 0;
    unsigned char header[2], msg[65535], record[40];
    setvbuf(stdin, NULL, _IOFBF, 1048576);
    for (;;) {
        size_t got = fread(header, 1, 2, stdin);
        bytes += got;
        if (got != 2) { truncated = got != 0; break; }
        unsigned len = (unsigned)be(header, 2);
        if (!len) { fprintf(stderr, "invalid zero-length ITCH message\n"); return 1; }
        got = fread(msg, 1, len, stdin);
        bytes += got;
        if (got != len) { truncated = 1; break; }
        unsigned char type = msg[0];
        if (len < min_length(type)) {
            fprintf(stderr, "short framed ITCH message: type %c length %u\n", type, len);
            return 1;
        }
        ++counts[type]; ++total;
        if (len >= 11) {
            uint64_t ts = be(msg + 5, 6);
            if (!have_ts) first_ts = ts;
            last_ts = ts;
            have_ts = 1;
        }
        if (len < 3) continue;
        unsigned locate = (unsigned)be(msg + 1, 2);
        if (type == 'R') {
            char name[9];
            memcpy(name, msg + 11, 8); name[8] = '\0';
            for (int i = 7; i >= 0 && name[i] == ' '; --i) name[i] = '\0';
            locates[locate] = -1;
            for (size_t i = 0; i < n_symbols; ++i)
                if (!strcmp(name, symbols[i].name)) { locates[locate] = (int)i; break; }
        }
        int index = locates[locate];
        if (index < 0) continue;
        Symbol *s = symbols + index;
        ++s->counts[type];
        if (!strchr("AFECXDUPQH", type) || type == 0) continue;
        memset(record, 0, sizeof(record));
        le(record, be(msg + 5, 6), 8);
        record[32] = type;
        switch (type) {
            case 'A': case 'F': case 'P':
                le(record + 8, be(msg + 11, 8), 8);
                record[33] = msg[19];
                le(record + 24, be(msg + 20, 4), 4);
                le(record + 28, be(msg + 32, 4), 4);
                break;
            case 'E': case 'C': case 'X':
                le(record + 8, be(msg + 11, 8), 8);
                le(record + 24, be(msg + 19, 4), 4);
                if (type == 'C') {
                    record[34] = msg[31];
                    le(record + 28, be(msg + 32, 4), 4);
                }
                break;
            case 'D': le(record + 8, be(msg + 11, 8), 8); break;
            case 'U':
                le(record + 8, be(msg + 11, 8), 8);
                le(record + 16, be(msg + 19, 8), 8);
                le(record + 24, be(msg + 27, 4), 4);
                le(record + 28, be(msg + 31, 4), 4);
                break;
            case 'Q': {
                uint64_t shares = be(msg + 11, 8);
                if (shares > UINT32_MAX) {
                    fprintf(stderr, "Q shares exceed uint32 record capacity\n"); return 1;
                }
                le(record + 24, shares, 4);
                le(record + 28, be(msg + 27, 4), 4);
                break;
            }
            case 'H': break; /* State/reason have no slot in the prescribed record. */
        }
        if (fwrite(record, 1, sizeof(record), s->out) != sizeof(record)) {
            perror("write record"); return 1;
        }
        ++s->records;
    }
    if (ferror(stdin)) { perror("read stdin"); return 1; }
    for (size_t i = 0; i < n_symbols; ++i)
        if (fclose(symbols[i].out)) { perror("close bin"); return 1; }
    snprintf(path, path_size, "%s/filter_counts.json", argv[1]);
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return 1; }
    fprintf(f, "{\n\"bytes_read\":%" PRIu64 ",\n\"total_messages\":%" PRIu64
            ",\n\"input_ended_mid_message\":%s,\n\"counts_by_type\":",
            bytes, total, truncated ? "true" : "false");
    counts_json(f, counts);
    if (have_ts)
        fprintf(f, ",\n\"first_timestamp_ns\":%" PRIu64 ",\n\"last_timestamp_ns\":%" PRIu64,
                first_ts, last_ts);
    else fputs(",\n\"first_timestamp_ns\":null,\n\"last_timestamp_ns\":null", f);
    fputs(",\n\"symbols\":{", f);
    for (size_t i = 0; i < n_symbols; ++i) {
        fprintf(f, "%s\n\"%s\":{\"records\":%" PRIu64 ",\"counts_by_type\":",
                i ? "," : "", symbols[i].name, symbols[i].records);
        counts_json(f, symbols[i].counts);
        fputc('}', f);
    }
    fputs("\n}}\n", f);
    if (ferror(f) || fclose(f)) { perror("write counts"); return 1; }
    fprintf(stderr, "filter: %" PRIu64 " complete messages, %" PRIu64 " bytes, truncated=%d\n",
            total, bytes, truncated);
    free(path); free(names); free(symbols);
    return 0;
}
