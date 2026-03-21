import time
import concurrent.futures
from main_pipeline import fact_check

# ================= TEST CLAIMS =================
TEST_CLAIMS = [
    "Vaccines cause autism",
    "Earth is flat",
    "India won the 2011 Cricket World Cup",
    "Neil Armstrong never went to the moon",
    "COVID-19 was created in a lab",
    "The Eiffel Tower is in London",
    "Albert Einstein failed mathematics",
    "Drinking water cures cancer",
    "5G towers spread coronavirus",
    "Climate change is not real",
]

# ================= SINGLE BENCHMARK =================
def benchmark_single():
    print("\nSingle claim benchmark...")
    print("="*50)

    times = []
    for claim in TEST_CLAIMS[:5]:
        start  = time.time()
        result = fact_check(claim)
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"Claim    : {claim[:50]}")
        print(f"Verdict  : {result.get('verdict', 'N/A')}")
        print(f"Time     : {round(elapsed, 2)}s")
        print("-"*50)

    avg = sum(times) / len(times)
    print(f"Avg time per claim : {round(avg, 2)}s")
    print(f"Est. throughput    : {round(60/avg, 1)} posts/min (single)")

# ================= BATCH BENCHMARK =================
def benchmark_batch(max_workers=10):
    print(f"\nBatch benchmark (workers={max_workers})...")
    print("="*50)

    claims  = TEST_CLAIMS * 5   # 50 claims total
    start   = time.time()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fact_check, claim): claim
            for claim in claims
        }
        for future in concurrent.futures.as_completed(futures):
            claim  = futures[future]
            try:
                result = future.result()
                results.append({
                    "claim"  : claim,
                    "verdict": result.get("verdict", "N/A")
                })
            except Exception as e:
                results.append({
                    "claim"  : claim,
                    "verdict": f"ERROR: {str(e)}"
                })

    elapsed    = time.time() - start
    throughput = len(claims) / elapsed * 60

    print(f"Total claims       : {len(claims)}")
    print(f"Time taken         : {round(elapsed, 2)}s")
    print(f"Throughput         : {round(throughput, 1)} posts/minute")
    print(f"Workers used       : {max_workers}")

    return throughput

# ================= WORKER SCALING TEST =================
def benchmark_scaling():
    print("\nScaling test — finding best worker count...")
    print("="*50)

    best_throughput = 0
    best_workers    = 1

    for workers in [1, 5, 10, 20, 50]:
        print(f"\nTesting {workers} workers...")
        throughput = benchmark_batch(max_workers=workers)

        if throughput > best_throughput:
            best_throughput = throughput
            best_workers    = workers

    print("\n" + "="*50)
    print(f"Best workers    : {best_workers}")
    print(f"Best throughput : {round(best_throughput, 1)} posts/minute")
    print("="*50)

# ================= RUN =================
if __name__ == "__main__":
    print("FACT CHECKER BENCHMARK")
    print("="*50)

    benchmark_single()
    benchmark_batch(max_workers=10)
    benchmark_scaling()

    print("\nBenchmark complete!")
    print("Use throughput number in your Intel Unnati submission!")