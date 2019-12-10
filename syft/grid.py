from collections import Counter
from typing import Any
from typing import Tuple
from typing import Counter as CounterType
from typing import Dict
from typing import Union


class VirtualGrid:
    def __init__(self, *workers):
        self.workers = workers

    def search(
        self, *query, verbose: bool = True, return_counter: bool = True
    ) -> Union[Tuple[Dict[Any, Any], CounterType], Dict[Any, Any]]:
        """Searches over a collection of workers, returning pointers to the results
        grouped by worker.
        """

        tag_counter: CounterType[int] = Counter()
        result_counter = 0

        results = {}
        for worker in self.workers:

            worker_tag_ctr: CounterType[int] = Counter()

            worker_results = worker.search(query)

            if len(worker_results) > 0:
                results[worker.id] = worker_results

                for result in worker_results:
                    for tag in result.tags:
                        tag_counter[tag] += 1
                        worker_tag_ctr[tag] += 1

                if verbose:
                    tags = str(worker_tag_ctr.most_common(3))
                    print(f"Found {str(len(worker_results))} results on {str(worker)} - {tags}")

                result_counter += len(worker_results)

        if verbose:
            print("\nFound " + str(result_counter) + " results in total.")
            print("\nTag Profile:")
            for tag, count in tag_counter.most_common():
                print("\t" + tag + " found " + str(count))

        if return_counter:
            return results, tag_counter
        else:
            return results
