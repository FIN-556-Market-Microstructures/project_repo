// itch_parser.cpp
//
// Nasdaq TotalView-ITCH 5.0 extractor for pcap.zst captures.
// C++ port of itch_parser.py. Same CLI, same CSV columns, same output,
// so the two can be diffed against each other on a sample file.
//
// Build:
//   g++ -O3 -march=native -std=c++17 itch_parser.cpp -lzstd -o itch_parser
//
// Run:
//   ./itch_parser path/to/xnas_pcaps
//   ./itch_parser path/to/xnas_pcaps --tickers AAPL,MSFT --start 20250203 --end 20250207
//   ./itch_parser path/to/xnas_pcaps --tickers ALL --split-by-date
//
// Several files at once (the parser is single threaded on purpose):
//   ls -d pcaps/2025*/ | xargs -P 8 -I{} ./itch_parser {} --outdir out

#include <zstd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------- byte helpers

static inline uint16_t be16(const uint8_t* p) { return (uint16_t(p[0]) << 8) | p[1]; }

static inline uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}

static inline uint64_t be48(const uint8_t* p) {
    return (uint64_t(p[0]) << 40) | (uint64_t(p[1]) << 32) | (uint64_t(p[2]) << 24) |
           (uint64_t(p[3]) << 16) | (uint64_t(p[4]) << 8) | p[5];
}

static inline uint64_t be64(const uint8_t* p) {
    return (uint64_t(be32(p)) << 32) | be32(p + 4);
}

static inline uint32_t le32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

// --------------------------------------------------------------------------- calendar
// Howard Hinnant's civil calendar algorithms, plus US Eastern DST rules.
// Avoids localtime_r (slow, locks, depends on the TZ database being present).

static long long days_from_civil(long long y, unsigned m, unsigned d) {
    y -= m <= 2;
    const long long era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = unsigned(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (long long)(doe) - 719468;
}

struct Civil {
    int year;
    unsigned month;
    unsigned day;
};

static Civil civil_from_days(long long z) {
    z += 719468;
    const long long era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = unsigned(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const long long y = (long long)(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;
    const unsigned m = mp + (mp < 10 ? 3 : -9);
    return Civil{int(y + (m <= 2)), m, d};
}

// 0 = Sunday
static unsigned weekday_from_days(long long z) {
    return unsigned(z >= -4 ? (z + 4) % 7 : (z + 5) % 7 + 6);
}

// Seconds to add to UTC to get US Eastern wall clock, for a given UTC instant.
static int eastern_offset(long long utc_sec) {
    const Civil c = civil_from_days(utc_sec / 86400 - (utc_sec % 86400 < 0 ? 1 : 0));
    const int y = c.year;

    const long long mar1 = days_from_civil(y, 3, 1);
    const unsigned mar1_wd = weekday_from_days(mar1);
    const unsigned second_sun_mar = 1 + ((7 - mar1_wd) % 7) + 7;

    const long long nov1 = days_from_civil(y, 11, 1);
    const unsigned nov1_wd = weekday_from_days(nov1);
    const unsigned first_sun_nov = 1 + ((7 - nov1_wd) % 7);

    // DST starts 02:00 local standard (07:00 UTC), ends 02:00 local daylight (06:00 UTC).
    const long long start = days_from_civil(y, 3, second_sun_mar) * 86400 + 7 * 3600;
    const long long end = days_from_civil(y, 11, first_sun_nov) * 86400 + 6 * 3600;

    return (utc_sec >= start && utc_sec < end) ? -4 * 3600 : -5 * 3600;
}

// --------------------------------------------------------------------------- formatting

static inline void put2(char* p, unsigned v) {
    p[0] = char('0' + v / 10);
    p[1] = char('0' + v % 10);
}

static inline void put_n(char* p, uint64_t v, int width) {
    for (int i = width - 1; i >= 0; --i) {
        p[i] = char('0' + v % 10);
        v /= 10;
    }
}

// Output buffer. One big append-only buffer per CSV, flushed in chunks.
struct Buf {
    FILE* f = nullptr;
    std::vector<char> b;
    size_t n = 0;

    explicit Buf(FILE* fp, size_t cap = 1u << 22) : f(fp), b(cap) {}

    inline void need(size_t k) {
        if (n + k > b.size()) flush();
        if (k > b.size()) b.resize(k * 2);
    }
    inline void ch(char c) {
        need(1);
        b[n++] = c;
    }
    inline void raw(const char* s, size_t k) {
        need(k);
        memcpy(b.data() + n, s, k);
        n += k;
    }
    inline void str(const std::string& s) { raw(s.data(), s.size()); }
    inline void uint(uint64_t v) {
        char tmp[20];
        int i = 20;
        do {
            tmp[--i] = char('0' + v % 10);
            v /= 10;
        } while (v);
        raw(tmp + i, size_t(20 - i));
    }
    // Price is 4 implied decimals. Match Python's float repr: trim trailing
    // zeros but always keep at least one decimal digit.
    inline void price(uint32_t p) {
        uint(p / 10000);
        ch('.');
        uint32_t frac = p % 10000;
        char d[4];
        put_n(d, frac, 4);
        int len = 4;
        while (len > 1 && d[len - 1] == '0') --len;
        raw(d, size_t(len));
    }
    void flush() {
        if (n) {
            fwrite(b.data(), 1, n, f);
            n = 0;
        }
    }
    ~Buf() {
        flush();
        if (f) fclose(f);
    }
};

// --------------------------------------------------------------------------- order table
// Open addressing on the uint64 order reference. std::unordered_map is the
// usual reason a C++ feed handler ends up only twice as fast as Python.

struct LiveOrder {
    uint64_t ref;
    uint16_t locate;
    char side;
    uint32_t shares;
    uint32_t price;
};

class OrderTable {
  public:
    OrderTable() { reset(1u << 20); }

    void clear() { reset(slots_.size()); }

    LiveOrder* find(uint64_t ref) {
        size_t m = slots_.size() - 1, i = hash(ref) & m;
        while (true) {
            Slot& s = slots_[i];
            if (s.state == EMPTY) return nullptr;
            if (s.state == FULL && s.o.ref == ref) return &s.o;
            i = (i + 1) & m;
        }
    }

    void insert(const LiveOrder& o) {
        if ((used_ + dead_ + 1) * 10 > slots_.size() * 7) grow();
        size_t m = slots_.size() - 1, i = hash(o.ref) & m;
        while (true) {
            Slot& s = slots_[i];
            if (s.state != FULL) {
                if (s.state == DEAD) --dead_;
                s.state = FULL;
                s.o = o;
                ++used_;
                return;
            }
            if (s.o.ref == o.ref) {  // replace in place
                s.o = o;
                return;
            }
            i = (i + 1) & m;
        }
    }

    bool erase(uint64_t ref, LiveOrder* out = nullptr) {
        size_t m = slots_.size() - 1, i = hash(ref) & m;
        while (true) {
            Slot& s = slots_[i];
            if (s.state == EMPTY) return false;
            if (s.state == FULL && s.o.ref == ref) {
                if (out) *out = s.o;
                s.state = DEAD;
                --used_;
                ++dead_;
                return true;
            }
            i = (i + 1) & m;
        }
    }

    size_t size() const { return used_; }

  private:
    enum State : uint8_t { EMPTY = 0, FULL = 1, DEAD = 2 };
    struct Slot {
        State state = EMPTY;
        LiveOrder o{};
    };

    static inline uint64_t hash(uint64_t x) {  // splitmix64 finalizer
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    void reset(size_t cap) {
        slots_.assign(cap, Slot{});
        used_ = dead_ = 0;
    }

    void grow() {
        std::vector<Slot> old;
        old.swap(slots_);
        size_t cap = old.size();
        if (used_ * 10 > cap * 5) cap *= 2;  // only really grow if it's live data
        reset(cap);
        for (const Slot& s : old)
            if (s.state == FULL) insert(s.o);
    }

    std::vector<Slot> slots_;
    size_t used_ = 0, dead_ = 0;
};

// --------------------------------------------------------------------------- zstd stream

class ZstdReader {
  public:
    explicit ZstdReader(const std::string& path) {
        f_ = fopen(path.c_str(), "rb");
        if (!f_) throw std::runtime_error("cannot open " + path);
        d_ = ZSTD_createDCtx();
        in_.assign(ZSTD_DStreamInSize(), 0);
        zin_ = {in_.data(), 0, 0};
    }
    ~ZstdReader() {
        if (d_) ZSTD_freeDCtx(d_);
        if (f_) fclose(f_);
    }
    ZstdReader(const ZstdReader&) = delete;
    ZstdReader& operator=(const ZstdReader&) = delete;

    // Decompress up to cap bytes into dst. Returns 0 at end of stream.
    size_t fill(uint8_t* dst, size_t cap) {
        ZSTD_outBuffer out{dst, cap, 0};
        while (out.pos == 0) {
            if (zin_.pos == zin_.size) {
                size_t got = fread(in_.data(), 1, in_.size(), f_);
                if (got == 0) return 0;
                zin_ = {in_.data(), got, 0};
            }
            size_t rc = ZSTD_decompressStream(d_, &out, &zin_);
            if (ZSTD_isError(rc)) throw std::runtime_error(ZSTD_getErrorName(rc));
        }
        return out.pos;
    }

  private:
    FILE* f_ = nullptr;
    ZSTD_DCtx* d_ = nullptr;
    std::vector<char> in_;
    ZSTD_inBuffer zin_{};
};

// Sliding window over the decompressed stream so pcap records can straddle
// decompression chunks without any per-record allocation.
class Feeder {
  public:
    explicit Feeder(ZstdReader& z, size_t cap = 1u << 22) : z_(z), b_(cap) {}

    bool ensure(size_t k) {
        if (end_ - pos_ >= k) return true;
        if (pos_) {
            memmove(b_.data(), b_.data() + pos_, end_ - pos_);
            end_ -= pos_;
            pos_ = 0;
        }
        if (k > b_.size()) b_.resize(k * 2);
        while (end_ - pos_ < k) {
            size_t got = z_.fill(b_.data() + end_, b_.size() - end_);
            if (got == 0) break;
            end_ += got;
        }
        return end_ - pos_ >= k;
    }
    const uint8_t* at() const { return b_.data() + pos_; }
    void skip(size_t k) { pos_ += k; }

  private:
    ZstdReader& z_;
    std::vector<uint8_t> b_;
    size_t pos_ = 0, end_ = 0;
};

// --------------------------------------------------------------------------- message table

struct Spec {
    int len;
    const char* name;
};

static Spec spec_for(char t) {
    switch (t) {
        case 'S': return {12, "SYSTEM_EVENT"};
        case 'H': return {25, "TRADING_ACTION"};
        case 'R': return {39, "DIRECTORY"};
        case 'A': return {36, "ADD"};
        case 'F': return {40, "ADD_MPID"};
        case 'E': return {31, "EXECUTE"};
        case 'C': return {36, "EXECUTE_PRICE"};
        case 'X': return {23, "CANCEL"};
        case 'D': return {19, "DELETE"};
        case 'U': return {35, "REPLACE"};
        case 'P': return {44, "TRADE"};
        case 'Q': return {40, "CROSS_TRADE"};
        case 'B': return {19, "BROKEN_TRADE"};
        default: return {0, nullptr};
    }
}

// --------------------------------------------------------------------------- config

struct Args {
    std::string root;
    std::string outdir = "itch_output";
    std::string start, end;
    std::unordered_set<std::string> tickers;
    bool all_tickers = true;
    bool split_by_date = false;
    bool track_book = true;
    bool include_admin = false;
    long max_files = -1;
};

static void usage_and_exit(const char* argv0) {
    fprintf(stderr,
            "usage: %s ROOT [--tickers A,B|ALL] [--outdir DIR] [--start YYYYMMDD]\n"
            "          [--end YYYYMMDD] [--max-files N] [--split-by-date]\n"
            "          [--no-book] [--include-admin]\n",
            argv0);
    exit(2);
}

static Args parse_args(int argc, char** argv) {
    if (argc < 2) usage_and_exit(argv[0]);
    Args a;
    a.root = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string f = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage_and_exit(argv[0]);
            return argv[++i];
        };
        if (f == "--tickers") {
            std::string v = next();
            for (auto& c : v) c = char(toupper(c));
            if (v == "ALL") {
                a.all_tickers = true;
            } else {
                a.all_tickers = false;
                size_t p = 0;
                while (p <= v.size()) {
                    size_t q = v.find(',', p);
                    if (q == std::string::npos) q = v.size();
                    std::string t = v.substr(p, q - p);
                    if (!t.empty()) a.tickers.insert(t);
                    p = q + 1;
                }
            }
        } else if (f == "--outdir") {
            a.outdir = next();
        } else if (f == "--start") {
            a.start = next();
        } else if (f == "--end") {
            a.end = next();
        } else if (f == "--max-files") {
            a.max_files = atol(next().c_str());
        } else if (f == "--split-by-date") {
            a.split_by_date = true;
        } else if (f == "--no-book") {
            a.track_book = false;
        } else if (f == "--include-admin") {
            a.include_admin = true;
        } else {
            fprintf(stderr, "unknown option: %s\n", f.c_str());
            usage_and_exit(argv[0]);
        }
    }
    return a;
}

// --------------------------------------------------------------------------- state

static const char* kColumns =
    "date_et,time_et,capture_utc,seq,msg_type,ticker,locate,order_ref,side,shares,price,match_num,"
    "note\r\n";

struct Writers {
    const Args& args;
    std::map<std::string, std::unique_ptr<Buf>> bufs;

    explicit Writers(const Args& a) : args(a) {}

    Buf& get(const std::string& date_key) {
        const std::string key = args.split_by_date ? date_key : "all";
        auto it = bufs.find(key);
        if (it != bufs.end()) return *it->second;
        std::string name = std::string("itch_orders") + (args.all_tickers ? "_all_" : "_") + key +
                           ".csv";
        fs::path p = fs::path(args.outdir) / name;
        FILE* f = fopen(p.string().c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + p.string());
        auto b = std::make_unique<Buf>(f);
        b->raw(kColumns, strlen(kColumns));
        printf("    writing %s\n", p.string().c_str());
        Buf& ref = *b;
        bufs.emplace(key, std::move(b));
        return ref;
    }

    void flush_all() {
        for (auto& kv : bufs) kv.second->flush();
    }
};

struct SymbolTable {
    // locate -> symbol. Locate codes are reassigned daily, so this is cleared
    // whenever a new session is detected.
    std::vector<char> names;   // 9 bytes per locate, NUL terminated
    std::vector<uint8_t> len;  // 0 = unknown
    std::vector<uint8_t> keep; // 1 = this locate is in our ticker list

    SymbolTable() : names(65536 * 9, 0), len(65536, 0), keep(65536, 0) {}

    void clear() {
        std::fill(len.begin(), len.end(), 0);
        std::fill(keep.begin(), keep.end(), 0);
    }
    void set(uint16_t loc, const uint8_t* sym8, bool wanted) {
        int n = 8;
        while (n > 0 && sym8[n - 1] == ' ') --n;
        memcpy(&names[size_t(loc) * 9], sym8, size_t(n));
        names[size_t(loc) * 9 + size_t(n)] = 0;
        len[loc] = uint8_t(n);
        keep[loc] = wanted ? 1 : 0;
    }
    const char* get(uint16_t loc) const { return &names[size_t(loc) * 9]; }
};

struct Stats {
    long long msgs = 0, rows = 0, jumps = 0, lost = 0;
};

// Cached epoch-second -> formatted date/time, since the second only changes
// once a second but we format on every message.
struct TimeCache {
    long long sec = -1;
    char et_date[11] = {0};   // YYYY-MM-DD
    char et_key[9] = {0};     // YYYYMMDD
    char utc_prefix[20] = {0};// YYYY-MM-DDTHH:MM:SS

    void update(long long s) {
        if (s == sec) return;
        sec = s;

        const long long et = s + eastern_offset(s);
        long long days = et / 86400;
        long long rem = et % 86400;
        if (rem < 0) {
            rem += 86400;
            --days;
        }
        Civil c = civil_from_days(days);
        put_n(et_date, uint64_t(c.year), 4);
        et_date[4] = '-';
        put2(et_date + 5, c.month);
        et_date[7] = '-';
        put2(et_date + 8, c.day);
        et_date[10] = 0;
        memcpy(et_key, et_date, 4);
        memcpy(et_key + 4, et_date + 5, 2);
        memcpy(et_key + 6, et_date + 8, 2);
        et_key[8] = 0;

        long long udays = s / 86400;
        long long urem = s % 86400;
        if (urem < 0) {
            urem += 86400;
            --udays;
        }
        Civil u = civil_from_days(udays);
        put_n(utc_prefix, uint64_t(u.year), 4);
        utc_prefix[4] = '-';
        put2(utc_prefix + 5, u.month);
        utc_prefix[7] = '-';
        put2(utc_prefix + 8, u.day);
        utc_prefix[10] = 'T';
        put2(utc_prefix + 11, unsigned(urem / 3600));
        utc_prefix[13] = ':';
        put2(utc_prefix + 14, unsigned(urem % 3600 / 60));
        utc_prefix[16] = ':';
        put2(utc_prefix + 17, unsigned(urem % 60));
        utc_prefix[19] = 0;
    }
};

// --------------------------------------------------------------------------- main parse

struct Parser {
    const Args& args;
    Writers& out;
    SymbolTable syms;
    OrderTable orders;
    TimeCache tc;
    std::map<std::string, long long> counts;  // "TICKER|TYPE" -> n
    std::vector<std::string> warnings;

    Parser(const Args& a, Writers& w) : args(a), out(w) {}

    void reset_session() {
        syms.clear();
        orders.clear();
    }

    bool wanted_symbol(const uint8_t* sym8) const {
        if (args.all_tickers) return true;
        int n = 8;
        while (n > 0 && sym8[n - 1] == ' ') --n;
        return args.tickers.count(std::string((const char*)sym8, size_t(n))) > 0;
    }

    Stats run_file(const std::string& path, const std::string& stamp);
};

Stats Parser::run_file(const std::string& path, const std::string& stamp) {
    Stats st;
    ZstdReader z(path);
    Feeder fd(z);

    if (!fd.ensure(24)) {
        warnings.push_back(stamp + ": file too short, skipped");
        return st;
    }
    uint32_t magic = le32(fd.at());
    if (magic != 0xA1B23C4Du && magic != 0xA1B2C3D4u) {
        warnings.push_back(stamp + ": not a little-endian pcap, skipped");
        return st;
    }
    const uint32_t frac_scale = (magic == 0xA1B23C4Du) ? 1u : 1000u;  // ns vs us
    fd.skip(24);

    uint8_t session[10];
    bool have_session = false;
    uint64_t expected_seq = 0;
    bool have_expected = false;

    char note[64];

    while (fd.ensure(16)) {
        const uint8_t* h = fd.at();
        uint32_t sec = le32(h), frac = le32(h + 4), caplen = le32(h + 8);
        fd.skip(16);
        if (!fd.ensure(caplen)) break;
        const uint8_t* pkt = fd.at();
        const size_t pktlen = caplen;
        fd.skip(caplen);

        if (pktlen <= 20) continue;

        // Ethernet (optionally one VLAN tag) / IPv4 / UDP
        size_t ip = 14;
        uint16_t eth = be16(pkt + 12);
        if (eth == 0x8100) {
            ip = 18;
            eth = be16(pkt + 16);
        }
        if (eth != 0x0800) continue;
        if (ip + 10 >= pktlen || pkt[ip + 9] != 17) continue;
        size_t off = ip + size_t(pkt[ip] & 0x0F) * 4 + 8;
        if (off + 20 > pktlen) continue;

        const uint8_t* pl = pkt + off;
        const size_t pllen = pktlen - off;

        if (!have_session || memcmp(pl, session, 10) != 0) {
            memcpy(session, pl, 10);
            if (have_session) {
                reset_session();
                printf("    MoldUDP64 session changed: tracking state reset\n");
            }
            have_session = true;
            have_expected = false;
        }

        uint64_t seq = be64(pl + 10);
        uint16_t count = be16(pl + 18);
        if (count == 0xFFFF) continue;  // heartbeat / end of session
        if (have_expected && seq != expected_seq) {
            ++st.jumps;
            if (seq > expected_seq) st.lost += (long long)(seq - expected_seq);
        }
        expected_seq = seq + count;
        have_expected = true;

        tc.update((long long)sec);
        const uint64_t nsec = uint64_t(frac) * frac_scale;

        size_t pos = 20;
        for (uint16_t k = 0; k < count; ++k) {
            if (pos + 2 > pllen) break;
            uint16_t mlen = be16(pl + pos);
            if (pos + 2 + mlen > pllen) break;
            const uint8_t* m = pl + pos + 2;
            pos += 2 + mlen;
            if (mlen < 11) continue;

            ++st.msgs;
            const char type = char(m[0]);
            const uint16_t locate = be16(m + 1);

            // Stock Directory: this is where locate codes get their symbol.
            if (type == 'R') {
                if (mlen >= 19) {
                    const uint8_t* sym = m + 11;
                    syms.set(locate, sym, wanted_symbol(sym));
                }
                continue;
            }

            const Spec sp = spec_for(type);
            if (!sp.name) continue;

            const bool admin = (type == 'S' || type == 'H');

            // Messages that carry the symbol inline let us learn a locate we
            // missed (file starts mid-session, no R messages seen).
            if ((type == 'A' || type == 'F' || type == 'P') && !syms.len[locate] && mlen >= 32) {
                const uint8_t* sym = m + 24;
                syms.set(locate, sym, wanted_symbol(sym));
            }

            if (!admin) {
                if (!syms.len[locate]) continue;        // unknown symbol
                if (!args.all_tickers && !syms.keep[locate]) continue;  // early out
            } else if (!args.include_admin) {
                continue;
            }

            if (int(mlen) != sp.len) {
                warnings.push_back(stamp + ": " + type + " message had unexpected length " +
                                   std::to_string(mlen));
                continue;
            }

            uint64_t ref = 0, match = 0;
            bool has_ref = false, has_match = false, has_shares = false, has_price = false;
            uint32_t shares = 0, price = 0;
            char side = 0;
            size_t note_len = 0;
            const char* ticker = admin ? "" : syms.get(locate);

            switch (type) {
                case 'A':
                case 'F': {
                    ref = be64(m + 11);
                    side = char(m[19]);
                    shares = be32(m + 20);
                    price = be32(m + 32);
                    has_ref = has_shares = has_price = true;
                    if (type == 'F') {
                        int n = 4;
                        while (n > 0 && m[36 + n - 1] == ' ') --n;
                        memcpy(note, m + 36, size_t(n));
                        note_len = size_t(n);
                    }
                    if (args.track_book) orders.insert(LiveOrder{ref, locate, side, shares, price});
                    break;
                }
                case 'D': {
                    ref = be64(m + 11);
                    has_ref = true;
                    if (args.track_book) {
                        LiveOrder o{};
                        if (orders.erase(ref, &o)) {
                            side = o.side;
                            price = o.price;
                            has_price = true;
                        }
                    }
                    break;
                }
                case 'E':
                case 'C': {
                    ref = be64(m + 11);
                    shares = be32(m + 19);
                    match = be64(m + 23);
                    has_ref = has_shares = has_match = true;
                    if (type == 'C') {
                        price = be32(m + 32);
                        has_price = true;
                        note_len = size_t(snprintf(note, sizeof note, "printable=%c", char(m[31])));
                    }
                    if (args.track_book) {
                        if (LiveOrder* o = orders.find(ref)) {
                            side = o->side;
                            if (type == 'E') {
                                price = o->price;
                                has_price = true;
                            }
                            if (o->shares <= shares)
                                orders.erase(ref);
                            else
                                o->shares -= shares;
                        }
                    }
                    break;
                }
                case 'X': {
                    ref = be64(m + 11);
                    shares = be32(m + 19);
                    has_ref = has_shares = true;
                    if (args.track_book) {
                        if (LiveOrder* o = orders.find(ref)) {
                            side = o->side;
                            price = o->price;
                            has_price = true;
                            if (o->shares <= shares)
                                orders.erase(ref);
                            else
                                o->shares -= shares;
                        }
                    }
                    break;
                }
                case 'U': {
                    uint64_t old_ref = be64(m + 11);
                    ref = be64(m + 19);
                    shares = be32(m + 27);
                    price = be32(m + 31);
                    has_ref = has_shares = has_price = true;
                    note_len = size_t(snprintf(note, sizeof note, "replaces %llu",
                                               (unsigned long long)old_ref));
                    if (args.track_book) {
                        LiveOrder o{};
                        if (orders.erase(old_ref, &o)) side = o.side;
                        orders.insert(LiveOrder{ref, locate, side, shares, price});
                    }
                    break;
                }
                case 'P': {
                    ref = be64(m + 11);
                    side = char(m[19]);
                    shares = be32(m + 20);
                    price = be32(m + 32);
                    match = be64(m + 36);
                    has_ref = has_shares = has_price = has_match = true;
                    break;
                }
                case 'Q': {
                    shares = uint32_t(be64(m + 11));
                    price = be32(m + 27);
                    match = be64(m + 31);
                    has_shares = has_price = has_match = true;
                    note_len = size_t(snprintf(note, sizeof note, "cross=%c", char(m[39])));
                    break;
                }
                case 'B': {
                    match = be64(m + 11);
                    has_match = true;
                    break;
                }
                case 'S': {
                    note_len = size_t(snprintf(note, sizeof note, "event=%c", char(m[11])));
                    break;
                }
                case 'H': {
                    int n = 4;
                    while (n > 0 && m[21 + n - 1] == ' ') --n;
                    note_len = size_t(snprintf(note, sizeof note, "trading_state=%c reason=%.*s",
                                               char(m[19]), n, (const char*)(m + 21)));
                    // trading action carries its own symbol
                    ticker = syms.len[locate] ? syms.get(locate) : "";
                    break;
                }
                default: break;
            }

            Buf& b = out.get(tc.et_key);

            // row
            b.raw(tc.et_date, 10);
            b.ch(',');
            {
                uint64_t ns = be48(m + 5);
                uint64_t s = ns / 1000000000ull, fr = ns % 1000000000ull;
                char t[18];
                put2(t, unsigned(s / 3600));
                t[2] = ':';
                put2(t + 3, unsigned(s % 3600 / 60));
                t[5] = ':';
                put2(t + 6, unsigned(s % 60));
                t[8] = '.';
                put_n(t + 9, fr, 9);
                b.raw(t, 18);
            }
            b.ch(',');
            b.raw(tc.utc_prefix, 19);
            b.ch('.');
            {
                char f9[9];
                put_n(f9, nsec, 9);
                b.raw(f9, 9);
            }
            b.ch('Z');
            b.ch(',');
            b.uint(seq + k);
            b.ch(',');
            b.raw(sp.name, strlen(sp.name));
            b.ch(',');
            b.raw(ticker, strlen(ticker));
            b.ch(',');
            b.uint(locate);
            b.ch(',');
            if (has_ref) b.uint(ref);
            b.ch(',');
            if (side) b.ch(side);
            b.ch(',');
            if (has_shares) b.uint(shares);
            b.ch(',');
            if (has_price) b.price(price);
            b.ch(',');
            if (has_match) b.uint(match);
            b.ch(',');
            if (note_len) b.raw(note, note_len);
            b.ch('\r');
            b.ch('\n');

            ++st.rows;
            std::string key = std::string(*ticker ? ticker : "(admin)") + "|" + sp.name;
            ++counts[key];
        }
    }
    return st;
}

// --------------------------------------------------------------------------- file discovery

struct FileEntry {
    std::string stamp;  // YYYYMMDDTHHMMSS
    std::string path;
};

static bool parse_stamp(const std::string& fname, std::string& out) {
    // expects ...YYYYMMDDTHHMMSS.pcap.zst
    const std::string suffix = ".pcap.zst";
    if (fname.size() < 15 + suffix.size()) return false;
    if (fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
    size_t e = fname.size() - suffix.size();
    if (e < 15) return false;
    const std::string s = fname.substr(e - 15, 15);
    if (s[8] != 'T') return false;
    for (int i = 0; i < 15; ++i) {
        if (i == 8) continue;
        if (!isdigit((unsigned char)s[i])) return false;
    }
    out = s;
    return true;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        std::vector<FileEntry> files;
        for (auto& e : fs::recursive_directory_iterator(args.root)) {
            if (!e.is_regular_file()) continue;
            std::string stamp;
            if (!parse_stamp(e.path().filename().string(), stamp)) continue;
            const std::string day = stamp.substr(0, 8);
            if (!args.start.empty() && day < args.start) continue;
            if (!args.end.empty() && day > args.end) continue;
            files.push_back({stamp, e.path().string()});
        }
        std::sort(files.begin(), files.end(),
                  [](const FileEntry& a, const FileEntry& b) { return a.stamp < b.stamp; });
        if (args.max_files > 0 && (long)files.size() > args.max_files)
            files.resize(size_t(args.max_files));
        if (files.empty()) {
            fprintf(stderr, "No *.pcap.zst files found under %s\n", args.root.c_str());
            return 1;
        }

        fs::create_directories(args.outdir);

        Writers writers(args);
        Parser parser(args, writers);

        long long total_msgs = 0, total_rows = 0;
        long long prev_epoch = -1;
        const auto t_all = std::chrono::steady_clock::now();

        for (size_t i = 0; i < files.size(); ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::string& stamp = files[i].stamp;

            // file start time, treated as UTC, only used for session-gap detection
            long long y = atoll(stamp.substr(0, 4).c_str());
            long long mo = atoll(stamp.substr(4, 2).c_str());
            long long d = atoll(stamp.substr(6, 2).c_str());
            long long hh = atoll(stamp.substr(9, 2).c_str());
            long long mi = atoll(stamp.substr(11, 2).c_str());
            long long ss = atoll(stamp.substr(13, 2).c_str());
            long long epoch = days_from_civil(y, unsigned(mo), unsigned(d)) * 86400 + hh * 3600 +
                              mi * 60 + ss;

            double mb = double(fs::file_size(files[i].path)) / 1e6;
            printf("[%zu/%zu] %s  %.1f MB\n", i + 1, files.size(), stamp.c_str(), mb);
            fflush(stdout);

            if (prev_epoch >= 0) {
                long long gap = epoch - prev_epoch;
                if (gap > 2 * 3600) {
                    parser.reset_session();
                    printf("    new trading session (gap of %llds): tracking state reset\n", gap);
                } else if (gap > 11 * 60) {
                    char w[160];
                    snprintf(w, sizeof w, "%s: about %lld file(s) missing just before this one",
                             stamp.c_str(), gap / 600 - 1);
                    parser.warnings.push_back(w);
                    printf("    WARNING: %s\n", w);
                }
            }
            prev_epoch = epoch;

            Stats st;
            try {
                st = parser.run_file(files[i].path, stamp);
            } catch (const std::exception& ex) {
                parser.warnings.push_back(stamp + ": " + ex.what() + "; file skipped");
                printf("    WARNING: %s\n", parser.warnings.back().c_str());
                continue;
            }

            total_msgs += st.msgs;
            total_rows += st.rows;
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("    %lld messages scanned, %lld rows kept, %.1fs%s\n", st.msgs, st.rows, secs,
                   st.jumps ? "  (see warnings)" : "");
            if (st.jumps) {
                char w[200];
                snprintf(w, sizeof w,
                         "%s: %lld sequence jump(s), roughly %lld message(s) missing from this feed",
                         stamp.c_str(), st.jumps, st.lost);
                parser.warnings.push_back(w);
            }
            writers.flush_all();
        }

        writers.flush_all();
        const double mins =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count() / 60.0;
        printf("\nDone: %zu file(s), %lld messages scanned, %lld rows in %zu CSV(s) (%.1f min)\n",
               files.size(), total_msgs, total_rows, writers.bufs.size(), mins);

        std::map<std::string, std::map<std::string, long long>> by_ticker;
        for (auto& kv : parser.counts) {
            size_t bar = kv.first.find('|');
            by_ticker[kv.first.substr(0, bar)][kv.first.substr(bar + 1)] = kv.second;
        }
        for (auto& kv : by_ticker) {
            printf("  %-8s ", kv.first.c_str());
            bool first = true;
            for (auto& t : kv.second) {
                printf("%s%s: %lld", first ? "" : ", ", t.first.c_str(), t.second);
                first = false;
            }
            printf("\n");
        }
        if (!parser.warnings.empty()) {
            printf("\nWarnings:\n");
            for (auto& w : parser.warnings) printf("  - %s\n", w.c_str());
        }
        return 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
