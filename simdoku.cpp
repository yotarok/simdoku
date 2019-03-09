#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <sys/resource.h>
#include <string.h>

#include <boost/preprocessor.hpp>

//#define USE_LOCKED_CANDIDATE 1

#if DISABLE_INLINE
#  define FORCE_INLINE
#else
#  define FORCE_INLINE __attribute__((always_inline)) inline
#endif

#ifndef ENABLE_MT
#  define ENABLE_MT 1
#endif
#define MT_IO_BUFSIZE_BASE 65536
#define DEFAULT_NTHREADS 32 // c4 xlarge has 32 vCPU cores


#if ENABLE_MT
#  include <pthread.h>
#  include <unistd.h>
#  include <sched.h>
#endif

#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <nmmintrin.h>

#define FATAL(...)  \
  do { std::fprintf(stderr, "FATAL: "); std::fprintf(stderr, __VA_ARGS__); std::exit(-1); } while(0)

constexpr uint8_t BOARDLEN = 81;
constexpr uint8_t STRIPLEN = 27;
constexpr uint8_t NUMROW = 9;
constexpr uint8_t NUMCOL = 9;
constexpr uint8_t NUMBLOCK = 9;
constexpr uint8_t BLOCKSIZE = 9;
constexpr uint8_t NDIGIT = 9;
constexpr uint16_t maskinit = (1 << 9) - 1;

const size_t blk2strips[9][3] = {
  {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
  {9, 12, 15}, {10, 13, 16}, {11, 14, 17},
  {18, 21, 24}, {19, 22, 25}, {20, 23, 26}
};

const size_t strip2block[27] = {
  0, 1, 2, 0, 1, 2, 0, 1, 2,
  3, 4, 5, 3, 4, 5, 3, 4, 5,
  6, 7, 8, 6, 7, 8, 6, 7, 8,
};

const size_t strip2belt[27] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 1, 1, 1, 1, 1,
  2, 2, 2, 2, 2, 2, 2, 2, 2,
};

#include "simdoku_utils.cpp"

const __m128i onesV16 = _mm_set1_epi16(1);
const __m128i zerosV = _mm_set1_epi16(0);
const __m128i maskinitV = _mm_set_epi16(0, 0, 0, 0,
                                        0, maskinit, maskinit, maskinit);

const __m128i clearV16[] = {
  _mm_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                0xFFFF, 0xFFFF, 0xFFFF, 0),
  _mm_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                0xFFFF, 0xFFFF, 0, 0xFFFF),
  _mm_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                0xFFFF, 0, 0xFFFF, 0xFFFF),
  _mm_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                0, 0xFFFF, 0xFFFF, 0xFFFF)
};
const __m128i mask_range = _mm_set_epi16(0, 0, 0, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF);
const __m128i not_mask_range = _mm_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0, 0, 0);
const __m128i board_range = _mm_set_epi16(0, 0xFFFF, 0xFFFF, 0xFFFF, 0, 0, 0, 0);
const __m128i not_board_range = _mm_set_epi16(0xFFFF, 0, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);


FORCE_INLINE uint64_t _lrotl(const uint64_t value, uint64_t shift)  {
  return (value << shift) | (value >> (sizeof(value)*8 - shift));
}


#define extract_mask(m, off) (_mm_extract_epi16 (m, off))
#define extract_board(m, off) (_mm_extract_epi16 (m, off+4))

template <int off>
FORCE_INLINE uint16_t extract_board_value(__m128i m) {
  uint16_t bb = extract_board(m, off);
  return bb ? (get_msb_pos(bb) + 1) : 0;
}

#define pack_mask(m2, m1, m0)       \
  _mm_set_epi16(0, 0, 0, 0, 0, m2, m1, m0);


FORCE_INLINE __m128i mask_reduce_or(__m128i x) {
  // Only mixes lower (mask) part (4 16-bits)
  __m128i y = x | _mm_shufflelo_epi16(x, 78); //0b01001110
  return y | _mm_shufflelo_epi16(y, 57); // 0b00111001
}

FORCE_INLINE uint16_t board2i16(char c) {
  return (c == '0') ? 0 : (1 << (c - '0' - 1));
}


#define board2mask(b) _mm_srli_si128(b, 8)
#define mask2board(m) _mm_slli_si128(m, 8)

#define is_any_unknown(b)                                               \
  (_mm_test_all_ones(~_mm_cmpeq_epi16(b | not_board_range, zerosV)) == 0)

#define is_invalid_mask(m)                                              \
  (_mm_test_all_ones(~_mm_cmpeq_epi16(m | board2mask(m), zerosV) | not_mask_range) == 0)

#define MOD_MASK(nstid, setter)                                         \
  if (! _mm_test_all_zeros(board[nstid] & setter, mask_range)) {        \
    board[nstid] &= ~(board[nstid] & setter);                           \
    dirty |= (1 << (nstid));                                            \
  }

#define is_pow_of_2_epi16(x)                                \
  (_mm_cmpeq_epi16(x, (x & _mm_add_epi16(~x, onesV16))) &   \
   (~_mm_cmpeq_epi16(x, zerosV)))

const __m128i popcnt_lookup_8 =
  _mm_setr_epi8 (0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
const __m128i low4_of_8 = _mm_set1_epi8 (0x0F);
const __m128i high8_of_16 = _mm_set1_epi16(0xFF00);
const __m128i low8_of_16 = _mm_set1_epi16(0x00FF);

#define popcnt_epi8(x)                                                  \
  _mm_add_epi8(_mm_shuffle_epi8(popcnt_lookup_8,                        \
                                _mm_and_si128(low4_of_8, x)),           \
               _mm_shuffle_epi8(popcnt_lookup_8,                        \
                                _mm_and_si128(low4_of_8,                \
                                              _mm_srli_epi16 (x, 4))))
FORCE_INLINE __m128i popcnt_epi16(__m128i x) {
  __m128i y = popcnt_epi8(x);
  return _mm_add_epi16(_mm_srli_epi16(y, 8), y & low8_of_16);
}

struct state {
  __m128i board[STRIPLEN];
  int nunk;
  uint32_t dirty;

  state() { }

  FORCE_INLINE uint32_t get_and_pop_dirty() {
    uint32_t ret = get_msb_pos(dirty);
    dirty &= ~(1 << ret);
    return ret;
  }

  state(const char* boardbuf) {
    nunk = 0;
    for (int n = 0, s = 0; n < 81; n += 3, ++ s) {
      board[s] =
        _mm_set_epi16(0,
                      board2i16(boardbuf[n+2]),
                      board2i16(boardbuf[n+1]),
                      board2i16(boardbuf[n+0]),
                      0, 0, 0, 0);
      if (boardbuf[n+0] == '0') ++ nunk;
      if (boardbuf[n+1] == '0') ++ nunk;
      if (boardbuf[n+2] == '0') ++ nunk;
    }
    initialize_mask();
  }

  void initialize_mask() {
    __m128i masks_zone[3], masks_row[NUMROW],  masks_blk[NUMBLOCK];
    for (int r = 0; r < NUMROW; ++ r) {
      masks_row[r] = maskinitV &
        ~(mask_reduce_or(board2mask(board[r*3+0]) |
                         board2mask(board[r*3+1]) |
                         board2mask(board[r*3+2])));
    }

    for (int z = 0; z < 3; ++ z) {
      masks_zone[z] = maskinitV &
        ~(board2mask(board[0*3 + z] | board[1*3 + z] | board[2*3 + z] |
                     board[3*3 + z] | board[4*3 + z] | board[5*3 + z] |
                     board[6*3 + z] | board[7*3 + z] | board[8*3 + z]));
    }

    for (int blk = 0; blk < NUMBLOCK; ++ blk) {
      masks_blk[blk] = maskinitV &
        ~(mask_reduce_or(board2mask(board[blk2strips[blk][0]]) |
                         board2mask(board[blk2strips[blk][1]]) |
                         board2mask(board[blk2strips[blk][2]])));
    }

    for (int r = 0, stid = 0; r < NUMROW; ++ r) {
      for (int z = 0; z < 3; ++ z, ++ stid) {
        int blkid = strip2block[stid];
        __m128i flag = _mm_cmpeq_epi16(board2mask(board[stid]), zerosV);
        board[stid] |=
          flag & (masks_row[r] & masks_zone[z] & masks_blk[blkid]);
        // ^ it's safe to skip mask_range mask because first 64B of flag is
        //   0 and all component masks also has first 64B of 0s
      }
    }
  }


  FORCE_INLINE bool fill_trivials() {
    int sid;
    __m128i cur, flag, setter, setter_mix;

    while (dirty) {

#define LOOP_BODY(x, i, unused)                                         \
      sid = get_and_pop_dirty();                                        \
      cur = board[sid];                                                 \
      if (is_invalid_mask(cur))  return false;                          \
                                                                        \
      flag = is_pow_of_2_epi16(cur);                                    \
      if (! _mm_test_all_zeros(flag, mask_range)) {                     \
        update_board(sid, flag, cur);                                   \
        if (nunk == 0) return true;                                     \
        int row = sid / 3, zid = sid % 3; /* DIV op should be used */   \
        setter = cur & flag & mask_range;                               \
        setter_mix = mask_reduce_or(setter);                            \
        update_mask_with_setter(sid, row, zid,                          \
                                setter, setter_mix);                    \
      }                                                                 \
      if (dirty == 0) break;

      BOOST_PP_REPEAT(2, LOOP_BODY, ~);
#undef LOOP_BODY
    }
    return true;
  }

  FORCE_INLINE void update_board1(int sid, int off, uint16_t probe) {
    __m128i setter = _mm_set_epi64x((uint64_t)probe << (16 * off),
                                    0);
    board[sid] = (board[sid] | setter) & clearV16[off];
    -- nunk;

    setter = board2mask(setter);
    __m128i setter_mix = mask_reduce_or(setter);
    update_mask_with_setter(sid, sid / 3, sid % 3,
                            setter, setter_mix);
  }

  FORCE_INLINE void update_board(int sid, __m128i flag, __m128i mask) {
    // update board with flag and mask
    //board[sid] |= mask2board(mask & flag);
    // clear mask
    //board[sid] &= (~flag | not_mask_range);
    board[sid] = (board[sid] | mask2board(mask & flag)) & (~flag | not_mask_range);
    nunk -= _mm_popcnt_u64(_mm_extract_epi64(flag & onesV16, 0));
  }

  FORCE_INLINE
  void update_mask_with_setter(int sid, int row, int zid,
                               __m128i setter, __m128i setter_mix) {
    __m128i x;
    int sid_rowbegin = row * 3;
    int beltid = strip2belt[sid]; //row / 3;
    int row_blkbegin = beltid * 3;
    int rowoff_inblk = row - row_blkbegin;


    // Propagate horizontally
    // (include self modification here for maximize cache hit prob)
    if (zid == 0) {
      MOD_MASK(sid, setter_mix);
      MOD_MASK(sid_rowbegin+1, setter_mix);
      MOD_MASK(sid_rowbegin+2, setter_mix);
    } else if (zid == 1) {
      MOD_MASK(sid_rowbegin+0, setter_mix);
      MOD_MASK(sid, setter_mix);
      MOD_MASK(sid_rowbegin+2, setter_mix);
    } else if (zid == 2) {
      MOD_MASK(sid_rowbegin+0, setter_mix);
      MOD_MASK(sid_rowbegin+1, setter_mix);
      MOD_MASK(sid, setter_mix);
    }

    // Propagate within block
    if (rowoff_inblk == 0) {
      MOD_MASK(sid+3, setter_mix);
      MOD_MASK(sid+6, setter_mix);
    } else if (rowoff_inblk == 1) {
      MOD_MASK(sid-3, setter_mix);
      MOD_MASK(sid+3, setter_mix);
    } else if (rowoff_inblk == 2) {
      MOD_MASK(sid-3, setter_mix);
      MOD_MASK(sid-6, setter_mix);
    }

    // Propagate vertically
    if (beltid != 0) {
      MOD_MASK(0 + zid, setter);
      MOD_MASK(3 + zid, setter);
      MOD_MASK(6 + zid, setter);
    }
    if (beltid != 1) {
      MOD_MASK(9 + zid, setter);
      MOD_MASK(12 + zid, setter);
      MOD_MASK(15 + zid, setter);
    }
    if (beltid != 2) {
      MOD_MASK(18 + zid, setter);
      MOD_MASK(21 + zid, setter);
      MOD_MASK(24 + zid, setter);
    }
  }


  FORCE_INLINE bool fill_complementarity() {
    __m128i self, setter_oth;
    __m128i flag, nmask, setter, omask;
    __m128i setter_mix;
    int sid_rowbegin, belt, row_in_block;

#define LOOP_BODY(x,z,r)                                                \
    omask =                                                             \
      ((r == 0) ? zerosV : board[0 * 3 + z]) |                          \
      ((r == 1) ? zerosV : board[1 * 3 + z]) |                          \
      ((r == 2) ? zerosV : board[2 * 3 + z]) |                          \
      ((r == 3) ? zerosV : board[3 * 3 + z]) |                          \
      ((r == 4) ? zerosV : board[4 * 3 + z]) |                          \
      ((r == 5) ? zerosV : board[5 * 3 + z]) |                          \
      ((r == 6) ? zerosV : board[6 * 3 + z]) |                          \
      ((r == 7) ? zerosV : board[7 * 3 + z]) |                          \
      ((r == 8) ? zerosV : board[8 * 3 + z]);                           \
    self = board[sid_rowbegin + z];                                     \
    nmask = self & ~(omask);                                            \
    flag = is_pow_of_2_epi16(nmask);                                    \
    if (! _mm_test_all_zeros(flag, mask_range)) {                       \
      update_board(sid_rowbegin + z, flag, nmask);                      \
      if (nunk == 0) return true;                                       \
      setter = flag & nmask & mask_range;                               \
      setter_mix = mask_reduce_or(setter);                              \
      update_mask_with_setter(sid_rowbegin + z, r, z,                   \
                              setter, setter_mix);                      \
      self = board[sid_rowbegin + z];                                   \
      if (is_invalid_mask(self)) return false;                          \
    }                                                                   \
    setter_oth =                                                        \
      _mm_shufflelo_epi16(self, 9) |                                    \
      _mm_shufflelo_epi16(self, 18);                                    \
    if (z == 0)                                                         \
      omask = mask_reduce_or(board[sid_rowbegin+1] | board[sid_rowbegin+2]); \
    else if (z == 1)                                                    \
      omask = mask_reduce_or(board[sid_rowbegin+0] | board[sid_rowbegin+2]); \
    else if (z == 2)                                                    \
      omask = mask_reduce_or(board[sid_rowbegin+0] | board[sid_rowbegin+1]); \
    omask |= setter_oth;                                                \
    nmask = self & ~(omask);                                            \
    flag = is_pow_of_2_epi16(nmask);                                    \
    if (! _mm_test_all_zeros(flag, mask_range)) {                       \
      update_board(sid_rowbegin + z, flag, nmask);                      \
      if (nunk == 0) return true;                                       \
      setter = flag & nmask & mask_range;                               \
      setter_mix = mask_reduce_or(setter);                              \
      update_mask_with_setter(sid_rowbegin + z, r, z,                   \
                              setter, setter_mix);                      \
      self = board[sid_rowbegin + z];                                   \
      if (is_invalid_mask(self)) return false;                          \
      setter_oth =                                                      \
        _mm_shufflelo_epi16(self, 9) |                                  \
        _mm_shufflelo_epi16(self, 18);                                  \
    }                                                                   \
    if (row_in_block == 0) {                                            \
      omask = mask_reduce_or(board[sid_rowbegin+z+3] | board[sid_rowbegin+z+6]); \
    } else if (row_in_block == 1) {                                     \
      omask = mask_reduce_or(board[sid_rowbegin+z-3] | board[sid_rowbegin+z+3]); \
    } else if (row_in_block == 2) {                                     \
      omask = mask_reduce_or(board[sid_rowbegin+z-6] | board[sid_rowbegin+z-3]); \
    }                                                                   \
    omask |= setter_oth;                                                \
    nmask = self & ~(omask);                                            \
    flag = is_pow_of_2_epi16(nmask);                                    \
    if (! _mm_test_all_zeros(flag, mask_range)) {                       \
      update_board(sid_rowbegin + z, flag, nmask);                      \
      if (nunk == 0) return true;                                       \
      setter = flag & nmask & mask_range;                               \
      setter_mix = mask_reduce_or(setter);                              \
      update_mask_with_setter(sid_rowbegin + z, r, z,                   \
                              setter, setter_mix);                      \
    }

    int sid = 0;
    sid_rowbegin = 0; belt = 0; row_in_block = 0;
    BOOST_PP_REPEAT(3,LOOP_BODY,0);
    sid_rowbegin = 3; belt = 0; row_in_block = 1;
    BOOST_PP_REPEAT(3,LOOP_BODY,1);
    sid_rowbegin = 6; belt = 0; row_in_block = 2;
    BOOST_PP_REPEAT(3,LOOP_BODY,2);
    sid_rowbegin = 9; belt = 1; row_in_block = 0;
    BOOST_PP_REPEAT(3,LOOP_BODY,3);
    sid_rowbegin = 12; belt = 1; row_in_block = 1;
    BOOST_PP_REPEAT(3,LOOP_BODY,4);
    sid_rowbegin = 15; belt = 1; row_in_block = 2;
    BOOST_PP_REPEAT(3,LOOP_BODY,5);
    sid_rowbegin = 18; belt = 2; row_in_block = 0;
    BOOST_PP_REPEAT(3,LOOP_BODY,6);
    sid_rowbegin = 21; belt = 2; row_in_block = 1;
    BOOST_PP_REPEAT(3,LOOP_BODY,7);
    sid_rowbegin = 24; belt = 2; row_in_block = 2;
    BOOST_PP_REPEAT(3,LOOP_BODY,8);
#undef LOOP_BODY

    return true;

  }

  FORCE_INLINE bool fill_greedily() {
    while(dirty != 0) {
      if (! fill_trivials()) return false;
      if (nunk == 0) break;
      if (! fill_complementarity()) return false;
      if (nunk == 0) break;
    }
    return true;
  }


  void output_board(char* dest) const {
    for (int n = 0, s = 0; n < 81; n += 3, ++ s) {
      dest[n] = '0' + extract_board_value<0>(board[s]);
      dest[n+1] = '0' + extract_board_value<1>(board[s]);
      dest[n+2] = '0' + extract_board_value<2>(board[s]);
    }
    dest[81] = '\n';
  }
} __attribute__ ((aligned (64)));
// ^ Common multiple of SSE alignment (16) and cache line (64)

bool solve(char* boardstr) {
  bounded_stack<state, 81 * 9> stk;
  state init(boardstr);
  init.dirty = 0x7FFFFFF; // 27 bits of 1s

  stk.push(init);

  while (! stk.empty()) {
    auto st = stk.top();
    stk.pop();

    if (! st.fill_greedily()) {
      continue;
    }

    if (st.nunk == 0) {
      st.output_board(boardstr);
      return true;
    }

    int minamb = 10;
    int minsid = 0, minoff = 0;
    uint16_t amb = 0;
    uint64_t cnts;

    {
#pragma unroll
      for (int sid = 0; sid < 27; ++ sid) {
        cnts = _mm_extract_epi64(popcnt_epi16(st.board[sid]), 0);
#define LOOP_BODY(x, off, unused)                           \
        if (cnts == 0) continue;                            \
        amb = cnts & 0xFF;                                  \
        if (amb == 2 || amb == 3) { /* 3 is good enough */  \
          minsid = sid;                                     \
          minoff = off;                                     \
          goto endloop;                                     \
        } else if (amb != 0 && amb < minamb) {              \
          minamb = amb;                                     \
          minsid = sid;                                     \
          minoff = off;                                     \
        }                                                   \
        cnts >>= 16;
        BOOST_PP_REPEAT(3, LOOP_BODY, ~);
#undef LOOP_BODY
      }
    endloop:
      ;
    }

    uint16_t minmask = ((_mm_extract_epi64(st.board[minsid], 0) >> (minoff * 16)) & 0xFFFF);
    uint16_t probe = 1;
    for (uint8_t v = 1; v <= 9; ++ v, probe <<= 1) {
      if (0 == (minmask & probe)) continue;
      state& nst = stk.next();
      nst = st;
      nst.dirty = 0;
      nst.update_board1(minsid, minoff, probe);
      stk.push_next();
    }
  }

  std::fprintf(stderr, "No hypothesis remaining\n");
  init.output_board(boardstr);
  return false;
}


#if ENABLE_MT
struct thread_info {
  int thread_id;
  int nthreads;
  int* preadhead;
  int solverhead;
  char* boardbuf;
};

//#define IDLE() ::sleep(0)
#define IDLE() ::sched_yield()
#define BUFSTRIDE (BOARDLEN + 1)

void* thread_main(void* p) {
  thread_info* pinfo = (thread_info*) p;
  //std::fprintf(stderr, "Thread %d: Start\n", pinfo->thread_id);

  while (1) {
    //std::fprintf(stderr, "Thread %d: Wait reader for reading %d, currently at %d\n",
    //pinfo->thread_id, pinfo->solverhead, *(pinfo->preadhead));
    while (pinfo->solverhead >= *(pinfo->preadhead)) {
      IDLE();
    }
    //std::fprintf(stderr, "Thread %d: Solve %d (Offset = %d)\n",
    //pinfo->thread_id, pinfo->solverhead, (83 * pinfo->solverhead));
    solve(pinfo->boardbuf + (BUFSTRIDE * pinfo->solverhead));
    //std::fprintf(stderr, "Thread %d: done\n",
    //pinfo->thread_id);
    pinfo->solverhead += pinfo->nthreads;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  FILE* input = stdin, * output = stdout;
  if (argc >= 2) {
    input = std::fopen(argv[1], "r");
  }
  if (argc >= 3) {
    output = std::fopen(argv[2], "w");
  }

  int nthreads = DEFAULT_NTHREADS;
  if (::getenv("SUDOKU_NTHREADS")) {
    nthreads = ::atoi(::getenv("SUDOKU_NTHREADS"));
  }

  std::fprintf(stderr, "Thread-M: Run with %d threads\n", nthreads);
  size_t MT_IO_BUFSIZE = MT_IO_BUFSIZE_BASE * nthreads;
  char* boardbuf = (char*) std::malloc(MT_IO_BUFSIZE * BUFSTRIDE * sizeof(char));
  ::pthread_t* threads = (::pthread_t*) std::malloc(sizeof(::pthread_t) * nthreads);
  thread_info* info = (thread_info*) std::malloc(sizeof(thread_info) * nthreads);
  int readhead = 0;
  int writehead = 0;
  int nsolved = 0, nprocessed = 0;

  for (int n = 0; n < nthreads; ++ n) {
    info[n].thread_id = n;
    info[n].nthreads = nthreads;
    info[n].boardbuf = &(boardbuf[0]);
    info[n].preadhead = &readhead;
    info[n].solverhead = n;
    ::pthread_create(&(threads[n]), 0, thread_main,
                     reinterpret_cast<void*>(&(info[n])));
  }

  unsigned long timer_start = milli_time(), time;
  /// Main thread just read and write
  bool reading = true;
  while (reading) {
    while (readhead < MT_IO_BUFSIZE) {
      for (int n = 0; n < nthreads; ++ n) {
        info[n].solverhead = n;
      }
      //char* p = std::fgets(&(boardbuf[readhead * BUFSTRIDE + 0]), 83, input);
      size_t nread = std::fread(&(boardbuf[readhead * BUFSTRIDE]), BUFSTRIDE, 1, input);
      if (nread == 0) {
        reading = false;
        break;
      }
      //std::fprintf(stderr, "Thread-M: Read line at %d\n", readhead);
      if (boardbuf[readhead * BUFSTRIDE + 0] < '0' || '9' < boardbuf[readhead * BUFSTRIDE + 0]) {
        std::fprintf(stderr, "parse error at %d in \"%s\"\n", readhead, boardbuf + (readhead * BUFSTRIDE));
        goto error;
      }
      ++ nprocessed;
      ++ readhead;
    }
    //std::fprintf(stderr, "Thread-M: Read completed. Waiting solvers\n");
    // Synchronize and wait finishing
    while (1) {
      bool finished = true;
      int minsolverhead = info[0].solverhead;
      for (int n = 1; n < nthreads; ++ n) {
        if (minsolverhead > info[n].solverhead) {
          minsolverhead = info[n].solverhead;
        }
      }

      std::fwrite(boardbuf + (writehead * BUFSTRIDE), BUFSTRIDE, minsolverhead - writehead, output);
      nsolved += (minsolverhead - writehead);
      writehead = minsolverhead;
      if (minsolverhead >= MT_IO_BUFSIZE) break;
      if (! reading && minsolverhead >= readhead) break;
      IDLE();
    }
    readhead = 0;
    writehead = 0;
  }
  time = milli_time() - timer_start;
  std::fprintf(stderr, "Solved %d out of %d in %lu.%03lu seconds (CPU time).\n", nsolved,
               nprocessed,
               time / 1000, time % 1000);

  return 0;
 error:
  std::fprintf(stderr, "parse error\n");
  return -1;
}

#else
#define READCHUNK 16

int main(int argc, char* argv[]) {
  FILE* input = stdin, * output = stdout;
  if (argc >= 2) {
    input = std::fopen(argv[1], "r");
  }
  if (argc >= 3) {
    output = std::fopen(argv[2], "w");
  }

  char boardbuf[READCHUNK * (BOARDLEN + 1) + 1];
  boardbuf[READCHUNK * (BOARDLEN + 1)] = '\0';
  // ^ Not used. Just for openining possibility to use fput/ fprintf.

  int nsolved = 0, nprocessed = 0;
  unsigned long timer_start = milli_time();

  while(1) {
    size_t read = std::fread(boardbuf, BOARDLEN + 1, READCHUNK, input);
    char* head = boardbuf;
    for (int i = 0; i < read; ++ i, head += (BOARDLEN + 1)) {
      if (*head < '0' || '9' < *head) break;
      ++ nprocessed;
      if (solve(head)) {
        ++ nsolved;
      } else {
      }
    }
    std::fwrite(boardbuf, BOARDLEN + 1, read, output);
    if (read < READCHUNK) break;
  }
  unsigned long time = milli_time() - timer_start;
  std::fprintf(stderr, "Solved %d out of %d in %lu.%03lu seconds.\n", nsolved, nprocessed,
               time / 1000, time % 1000);
  return 0;
}
#endif

