# TOOD: build this
import json
import numpy as np
  def _get_pnw(self, X):
        # calculate Pn(w) - probability distribution for negative sampling
        # basically just the word probability ^ 3/4
        word_freq = {}
        word_count = sum(len(x) for x in X)
        for x in X:
            for xj in x:
                if xj not in word_freq:
                    word_freq[xj] = 0
                word_freq[xj] += 1
        self.Pnw = np.zeros(self.V)
        for j in xrange(2, self.V): # 0 and 1 are the start and end tokens, we won't use those here
            self.Pnw[j] = (word_freq[j] / float(word_count))**0.75
        # print "self.Pnw[2000]:", self.Pnw[2000]
        assert(np.all(self.Pnw[2:] > 0))
        return self.Pnw

    def _get_negative_samples(self, context, num_neg_samples):
        # temporarily save context values because we don't want to negative sample these
        saved = {}
        for context_idx in context:
            saved[context_idx] = self.Pnw[context_idx]
            # print "saving -- context id:", context_idx, "value:", self.Pnw[context_idx]
            self.Pnw[context_idx] = 0
        neg_samples = np.random.choice(
            xrange(self.V),
            size=num_neg_samples, # this is arbitrary - number of negative samples to take
            replace=False,
            p=self.Pnw / np.sum(self.Pnw),
        )
        # print "saved:", saved
        for j, pnwj in saved.iteritems():
            self.Pnw[j] = pnwj
        assert(np.all(self.Pnw[2:] > 0))
        return neg_samples

