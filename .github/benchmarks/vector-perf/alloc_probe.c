#define _GNU_SOURCE
#include <malloc.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
typedef struct {
  uint64_t calls, bytes;
} Counts;
static _Thread_local int tracking;
static _Thread_local Counts counts;
static atomic_int all_tracking;
static atomic_ullong all_calls, all_bytes;
void vector_probe_begin_all(void) {
  atomic_store(&all_calls, 0);
  atomic_store(&all_bytes, 0);
  atomic_store(&all_tracking, 1);
}
Counts vector_probe_end_all(void) {
  atomic_store(&all_tracking, 0);
  return (Counts){atomic_load(&all_calls), atomic_load(&all_bytes)};
}
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);
extern void __libc_free(void *);
void vector_probe_begin(void) {
  counts = (Counts){0, 0};
  tracking = 1;
}
Counts vector_probe_end(void) {
  tracking = 0;
  return counts;
}
static void charge(size_t n) {
  if (tracking) {
    ++counts.calls;
    counts.bytes += n;
  }
  if (atomic_load_explicit(&all_tracking, memory_order_relaxed)) {
    atomic_fetch_add_explicit(&all_calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&all_bytes, n, memory_order_relaxed);
  }
}
void *malloc(size_t n) {
  charge(n);
  return __libc_malloc(n);
}
void *calloc(size_t n, size_t s) {
  charge(n * s);
  return __libc_calloc(n, s);
}
void *realloc(void *p, size_t n) {
  charge(n);
  return __libc_realloc(p, n);
}
void free(void *p) {
  __libc_free(p);
}
void *aligned_alloc(size_t a, size_t n) {
  charge(n);
  return __libc_memalign(a, n);
}
void *memalign(size_t a, size_t n) {
  charge(n);
  return __libc_memalign(a, n);
}
int posix_memalign(void **p, size_t a, size_t n) {
  if (a < sizeof(void *) || a % sizeof(void *) || (a & (a - 1))) return 22;
  charge(n);
  void *r = __libc_memalign(a, n);
  if (!r) return 12;
  *p = r;
  return 0;
}
