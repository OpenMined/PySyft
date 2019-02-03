from collections import Counter


class VirtualGrid:
    def __init__(self, *workers):
        self.workers = workers

    def search(self, *query):
        """Searches over a collection of workers, returning pointers to the results
        grouped by worker."""

        tag_ctr = Counter()
        result_ctr = 0

        results = {}
        for worker in self.workers:
            worker_results = worker.search(*query)
            for result in worker_results:
                for tag in result.tags:
                    tag_ctr[tag] += 1
            results[worker.id] = worker_results
            print("Found " + str(len(worker_results)) + " results on " + str(worker))
            result_ctr += len(worker_results)
        print("\nFound " + str(result_ctr) + " results in total.")
        print("\nTag Profile:")
        for tag, cnt in tag_ctr.most_common():
            print("\t" + tag + " found " + str(cnt))
        return results, tag_ctr
