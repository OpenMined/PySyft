"""
This class is an initial proof-of-concept implementation of PySyft's Differential Privacy
infrastructure. In particular, we want to provide the ability to automatically compute
the Sensitivity of a function. If you're unsure what Sensitivity or Differential Privacy
is, here are a couple of great resources:

- https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
- https://github.com/frankmcsherry/blog/blob/master/posts/2016-02-03.md

You can also find more information about Differential Privacy in the
#differential_privacy channel of OpenMined's slack.

This class keeps track of a per-user sensitivity and then returns the max sensitivity
across all users. Each user's sensitivity is intuitively a calculation of the maximum
amount (positively or negatively) that the output of a function would change if that
person were removed from the calculation. For example

If Bob had a variable:

    x = 5

and Alice had a variable

    y = 6

A function on these values might be

z = x + y

This tensor seeks to answer the question "what is the maximum amount z would change
if Bob or Alice were not included in the computation?". At first, one might say that
"the max for Bob is 5 and the max for Alice is 6." However, this is incorrect. In fact,
if we assume that Alice and Bob could have set their input variables to be *anything*
then the sensitivity is actually infinite.

Thus, by default, any value should have infinite sensitivity.

It is only by bounding the max/min possible value that we can track non-infinite sensitivity.
We can do this in a myriad of ways, but the easiest way is to simply use a threshold. Perhpas:

x =  max(min(5, 10), 0) # Bob's input

y =  max(min(6, 10), 0) # Alice's input

In this case, we would now know that Bob and Alice's numbers are both bounded to be between 0
and 10. Now we know that the following:

z = x + y

has a maximum sensitivity of 10 because the most the output could change by removing any user's
input is 10. Note that the max value z could actually take is 20, but this is different from
the max sensitivity.

The goal of this tensor is to make this kind of logic automatically happen on tensor objects so
that we can chan arbitrary tensor commands one after the other and always keep track of the
maximum amount that the output would change if ANY user were removed. This means we keep track
of the maximum positive and negative change any entity can have on any tensor value. In the case
that only one entity contributes to a variable (i.e., they initialized the variable), then the
min/max amount that entity contributes to their value also corresponds to the min/max value that
variable can take on.

We compute sensitivity because it's an important input to many important functions in Differential
Privacy.
"""


import torch as th

# TODO: determine max according to type(self.values)
max_long = 2 ** 62


class SensitivityTensor:
    def __init__(
        self, values: th.Tensor, max_vals: th.Tensor, min_vals: th.Tensor, entities: th.Tensor
    ):
        """Initializes sensitivity tensor.

        Args:
            values (torch.Tensor): the actual data points - can be any dimension
            max_vals (torch.Tensor): Theoretically we always interpret self.values as a sum over contributions from
                all possible known entities (which can be represented in a sparse matrix. TODO: test)
                Given this representation, self.max_vals can be thought of as the maximum amount
                that any entity could have added into self.values. Note, that if only one entity
                is contributing to a datapoint, then the max val of that entry in the tensor is
                also the maximum that entry in the tensor can take. self.max_vals should be the same
                shape as self.values with an extra dimension at the end which is as long as the
                number of known entities. TODO: add support for sparse recording of known entities

            min_vals (torch.Tensor): same as self.max_vals but instead of recording the maximum amount any entity
                adds to create self.values, it records the maximum amount any entity could be
                subtracting to create self.values.

            entities (torch.Tensor): a matrix of 1s and 0s which corresponds to whether or not a given entity
                was used to help encode a specific value. TODO: remove this tensor & sparsely encode max_vals/min_vals

        """

        self.values = values

        # to be clear, every entry in self.values has a list of corresponding entries, one for each entity.
        # Thus, self.max_vals and self.min_vals are the same shape as self.values with on added dimension
        # at the end of length #num entities (number of known entities - which must be known a-prior ATM)

        self.max_vals = max_vals  # (self.values.shape..., #num entities)
        self.min_vals = min_vals  # (self.values.shape..., #num entities)

        # one hot encoding of entities in the ancestry of this tensor
        self.entities = entities  # (self.values.shape..., #num entities)

    def __add__(self, other):
        """Adds the self tensor with a tensor or a scalar/tensor of equal shape, keeping track of sensitivity
        appropriately.

        Args:
            other (torch.Tensor, float, int, SensitivityTensor): the other tensor or scalar to add to this one.

        """

        # add to a private number
        if isinstance(other, SensitivityTensor):

            # perform addition on the raw data (nothing interesting to see here)
            new_vals = self.values + other.values

            # Perhaps the simplest sensitivity calculation. If two entities add two values together, then
            # the maximum value of the result is the sum of the maximum possible value of each of the inputs.
            # Note that if multiple entities contributed to each of the variables, then each entity's maximum
            # potential contribution is also summed.

            # Multiplication by .entities merely masks out entities which aren't being used.
            new_max_vals = (self.max_vals * self.entities) + (other.max_vals * other.entities)

            # Similarly, if two entities add two values together, then the minimum value that the result could
            # take on is simply the sum of the minimum values of the inputs. If multiple entities contributed
            # to each of the input variables, then each entitiy's maximum negative contribution is also summed

            # Multiplication by .entities merely masks out entities which aren't being used.
            new_min_vals = (self.min_vals * self.entities) + (other.min_vals * other.entities)

        else:
            # add to a public number
            new_vals = self.values + other

            # If the value of `other' is not a SensitivityTensor, then we assume that it is public and has a max_val
            # and min_val set at exactly the data, which we can use to sum with the current self.max_vals and
            # self.min_vals. We ignore self.entities because we can without meaningfully changing the result.
            new_max_vals = self.max_vals + other
            new_min_vals = self.min_vals + other

        return SensitivityTensor(values=new_vals, max_vals=new_max_vals, min_vals=new_min_vals)

    def __mul__(self, other):
        """Multiplies self by either another SensitivityTensor or a scalar or a tensor of equal size.

        Args:
            other (torch.Tensor, float, int, SensitivityTensor): the other tensor or scalar to multiply to this one.

        """

        # if the other tensor is a scalar TODO: add support for arbitrary torch tensors (which should work)
        if not isinstance(other, SensitivityTensor):

            # add to a public number
            new_vals = self.values * other

            # if the scalar is greater than 0, then we proceed as normal
            if other > 0:
                new_max_vals = self.max_vals * other
                new_min_vals = self.min_vals * other

            # if the scalar is negative, then it means it's going to flip the sign which means we need to flip
            # the position of min/max val as well because they're going to jump to the opposite sides of 0 (which means
            # for example, if other == -1, the max value would actually become the min and vise versa)
            else:
                new_min_vals = self.max_vals * other
                new_max_vals = self.min_vals * other

        # i the other tensor is a sensitivty tensor, then we must consider it's max/min values for each entity
        else:

            # just multiplying the values... nothing to see here
            new_vals = self.values * other.values

            #
            new_self_max_vals = th.max(
                self.min_vals * other.expanded_minminvals, self.max_vals * other.expanded_maxmaxvals
            )

            new_self_min_vals = th.min(
                self.min_vals * other.expanded_maxmaxvals, self.max_vals * other.expanded_minminvals
            )

            new_other_max_vals = th.max(
                other.min_vals * self.expanded_minminvals, other.max_vals * self.expanded_maxmaxvals
            )

            new_other_min_vals = th.max(
                other.min_vals * self.expanded_maxmaxvals, other.max_vals * self.expanded_minminvals
            )

            entities_self_or_other = (self.entities + other.entities) > 0

            new_self_max_vals = (new_self_max_vals * self.entities) + (
                (1 - self.entities) * -max_long
            )
            new_other_max_vals = (new_other_max_vals * self.entities) + (
                (1 - other.entities) * -max_long
            )

            new_max_vals = th.max(new_self_max_vals, new_other_max_vals) * entities_self_or_other

            new_self_min_vals = (new_self_min_vals * self.entities) + (
                (1 - self.entities) * max_long
            )
            new_other_min_vals = (new_other_min_vals * self.entities) + (
                (1 - other.entities) * max_long
            )

            new_min_vals = th.min(new_self_min_vals, new_other_min_vals) * entities_self_or_other

        return SensitivityTensor(new_vals, new_max_vals, new_min_vals)

    def __neg__(self):

        # note that new_min_vals and new_max_vals are reversed intentionally
        return SensitivityTensor(-self.values, -self.min_vals, -self.max_vals)

    def __sub__(self, other):

        # add to a private number
        if isinstance(other, SensitivityTensor):

            # just telling the data to do subtraction... nothing to see here
            new_vals = self.values - other.values

            # note that other.max/min values are reversed on purpose
            # because this functionality is equivalent to
            # output = self + (other * -1) and multiplication by
            # a negative number swaps the max/min values with each
            # other and flips their sign

            # similar to __add__ but with
            new_max_vals = (self.entities * self.max_vals) - (other.entities * other.min_vals)
            new_min_vals = (self.entities * self.min_vals) - (other.entities * other.max_vals)

        else:
            # add to a public number
            new_vals = self.values - other
            new_max_vals = self.max_vals - other
            new_min_vals = self.min_vals - other

        return SensitivityTensor(new_vals, new_max_vals, new_min_vals)

    def __truediv__(self, other):

        if isinstance(other, SensitivityTensor):
            raise Exception("probably best not to do this - it's gonna be inf a lot")

        new_vals = self.values / other
        new_max_vals = self.max_vals / other
        new_min_vals = self.min_vals / other

        return SensitivityTensor(new_vals, new_max_vals, new_min_vals)

    def __gt__(self, other):
        """BUG: the zero values mess this up"""
        if isinstance(other, SensitivityTensor):

            new_vals = self.values > other.values

            # if self is bigger than the biggest possible other
            if_left = (self.min_vals > other.expanded_maxmaxvals).float() * self.entities

            # if self is smaller than the smallest possible other
            if_right = (self.max_vals < other.expanded_minminvals).float() * self.entities

            # if self doesn't overlap with other at all
            if_left_or_right = if_left + if_right  # shouldn't have to check if this > 2 assuming
            # other's max is > other's min

            # if self does overlap with other
            new_self_max_vals = 1 - if_left_or_right

            # can't have a threshold output less than 0
            new_self_min_vals = if_left_or_right * 0

            # if other is bigger than the smallest possible self
            if_left = (other.min_vals > self.expanded_maxmaxvals).float() * other.entities

            # if other is smaller than the smallest possible self
            if_right = (other.max_vals < self.expanded_minminvals).float() * other.entities

            # if other and self don't overlap
            if_left_or_right = if_left + if_right  # shouldn't have to check if this > 2 assuming
            # other's max is > other's min

            # if other and self do overlap
            new_other_max_vals = 1 - if_left_or_right

            # the smallest possible result is 0
            new_other_min_vals = new_self_min_vals + 0

            # only contribute information from entities in ancestry
            new_self_max_vals = (new_self_max_vals * self.entities) + (
                (1 - self.entities) * -max_long
            )
            new_other_max_vals = (new_other_max_vals * self.entities) + (
                (1 - self.entities) * -max_long
            )

            # only contribute information from entities in ancestry
            new_self_min_vals = (new_self_min_vals * self.entities) + (
                (1 - self.entities) * max_long
            )
            new_other_min_vals = (new_other_min_vals * self.entities) + (
                (1 - self.entities) * max_long
            )

            entities_self_or_other = ((self.entities + other.entities) > 0).float()

            new_max_val = th.max(new_self_max_vals, new_other_max_vals) * entities_self_or_other
            new_min_val = th.min(new_self_min_vals, new_other_min_vals) * entities_self_or_other

        else:

            new_vals = self.values > other

            if_left = other <= self.max_vals
            if_right = other >= self.min_vals
            if_and = if_left * if_right

            new_max_val = if_and
            new_min_val = new_max_val * 0

        return SensitivityTensor(new_vals, new_max_val, new_min_val)

    def __lt__(self, other):
        return other.__gt__(self)

    def clamp_min(self, other):

        if isinstance(other, SensitivityTensor):
            raise Exception("Not implemented yet")

        new_min_val = self.min_vals.clamp_min(other)

        return SensitivityTensor(self.values.clamp_min(other), self.max_vals, new_min_val)

    def clamp_max(self, other):

        if isinstance(other, SensitivityTensor):
            raise Exception("Not implemented yet")

        entities = self.entities

        new_max_val = self.max_vals.clamp_max(other)

        return SensitivityTensor(self.values.clamp_max(other), new_max_val, self.min_vals)

    @property
    def maxmaxvals(self):
        """This returns the maximum possible value over all entities"""

        return (self.max_vals * self.entities).sum(1)

    @property
    def expanded_maxmaxvals(self):
        return self.maxmaxvals.unsqueeze(1).expand(self.max_vals.shape)

    @property
    def minminvals(self):
        """This returns the minimum possible values over all entities"""

        return (self.min_vals * self.entities).min(1)[0]

    @property
    def expanded_minminvals(self):
        return self.minminvals.unsqueeze(1).expand(self.min_vals.shape)

    @property
    def sensitivity(self):
        return (self.max_vals - self.min_vals).sum(1)

    @property
    def entities(self):
        return self._entities

    @entities.setter
    def entities(self, x):
        self._entities = x.float()

    def hard_sigmoid(self):
        return self.min(1).max(0)

    def hard_sigmoid_deriv(self, leak=0.01):
        return ((self < 1) * (self > 0)) + (self < 0) * leak - (self > 1) * leak
