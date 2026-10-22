#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
typedef struct {
  uint64_t calls, bytes;
} Counts;
static atomic_int go, done;
static void *worker(void *unused) {
  (void)unused;
  while (!atomic_load(&go)) sched_yield();
  void *p = malloc(97);
  assert(p);
  free(p);
  atomic_store(&done, 1);
  return NULL;
}
int main(void) {
  void (*begin)(void) = dlsym(RTLD_DEFAULT, "vector_probe_begin");
  Counts (*end)(void) = dlsym(RTLD_DEFAULT, "vector_probe_end");
  void (*begin_all)(void) = dlsym(RTLD_DEFAULT, "vector_probe_begin_all");
  Counts (*end_all)(void) = dlsym(RTLD_DEFAULT, "vector_probe_end_all");
  assert(begin && end && begin_all && end_all);
  begin();
  void *a = malloc(97), *b = calloc(3, 11), *c = NULL,
       *d = aligned_alloc(64, 384);
  a = realloc(a, 193);
  assert(posix_memalign(&c, 64, 320) == 0);
  assert(a && b && c && d);
  free(a);
  free(b);
  free(c);
  free(d);
  Counts counts = end();
  assert(counts.calls == 5 && counts.bytes == 1027);
  pthread_t thread;
  assert(pthread_create(&thread, NULL, worker, NULL) == 0);
  begin_all();
  atomic_store(&go, 1);
  while (!atomic_load(&done)) sched_yield();
  counts = end_all();
  assert(counts.calls == 1 && counts.bytes == 97);
  assert(pthread_join(thread, NULL) == 0);
}
