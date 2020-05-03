import torch
from torch import nn


class CRF(nn.Module):
    def __init__(self, num_labels, start_tag_id, end_tag_id, batch_first=True):
        # pylint: disable=invalid-name
        super().__init__()

        self.num_labels = num_labels
        self.START_TAG_ID = start_tag_id
        self.END_TAG_ID = end_tag_id
        self.batch_first = batch_first
        self.transitions = nn.Parameter(
            torch.empty(self.num_labels, self.num_labels))
        self._init_hidden()

    def _init_hidden(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        # no transitions allowed to the beginning of sentence
        self.transitions.data[:, self.START_TAG_ID] = -10000.
        # no transitions allowed from the beginning of sentence
        # NOTE: it's a trick to use tensor[r] instead of tensor[r, :]
        self.transitions.data[self.END_TAG_ID] = -10000.

    def forward(self, emissions, tags, mask=None):
        """Compute the negative log-likelihood. See `log_likelihood` method."""
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self,
                       emissions: torch.Tensor,
                       tags: torch.Tensor,
                       mask: torch.Tensor = None):
        """Compute the probability of a sequence of tags given a sequence of
        emissions scores.

        Args:
            `emissions` (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, num_labels) if batch_first is True,
                (seq_len, batch_size, num_labels) otherwise.
            `tags` (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
            `mask` (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.

        Returns:
            torch.Tensor: the log-likelihoods for each sequence in the batch.
                Shape of (batch_size,)
        """

        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        log_partition = self._compute_log_partition(emissions, mask=mask)

        return torch.sum(scores - log_partition)

    def _compute_scores(self, emissions: torch.Tensor, tags: torch.Tensor,
                        mask: torch.FloatTensor):
        """Compute the scores for a given batch of emissions with their tags.

        Args:
            `emissions` (torch.Tensor): (batch_size, seq_len, num_labels)
            `tags` (Torch.LongTensor): (batch_size, seq_len)
            `mask` (Torch.FloatTensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """

        batch_size, *_ = emissions.shape
        scores = []
        # In BERT, last_valid_idx is the idx for label '[SEP]' in each label sequence.
        last_valid_idx = mask.int().sum(1) - 1
        zero = torch.zeros(1).long().to(mask.device)
        mask = mask.index_put((torch.arange(batch_size), last_valid_idx), zero)

        # Iterate by batch
        # NOTE: actually, (for a sentence) transition score and emission socore,
        # can be calculated individually because they are just
        for batch_idx in range(batch_size):
            last_idx = last_valid_idx[batch_idx]
            left_side = tags[batch_idx, :last_idx]
            right_side = tags[batch_idx, 1:last_idx + 1]
            # Transition score
            t_score = torch.sum(self.transitions[left_side, right_side])
            # Don't calculate the emission score for '[CLS]' and '[SEP]'
            e_score = torch.sum(
                emissions[batch_idx,
                          torch.arange(1, last_idx
                                       ), tags[batch_idx, 1:last_idx]])

            score = t_score + e_score
            scores.append(score.unsqueeze(0))

        return torch.cat(scores)

    def _compute_log_partition(self, emissions, mask):
        """Compute the partition function in log-space using the forward-algorithm.

        Args:
            emissions (torch.Tensor): (batch_size, seq_len, num_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """

        batch_size, seq_length, _ = emissions.shape
        last_valid_idx = mask.int().sum(1) - 1
        zero = torch.zeros(1).long().to(mask.device)
        mask = mask.index_put((torch.arange(batch_size), last_valid_idx), zero)

        # NOTE: we don't need to calculate in the form of log_sum_exp
        # for the first valid token (not START token).
        # Don't worry about the sequence ['[CLS]', '[SEP]'] for adding emissions[:, 1]
        # into `alphas`, we don't predict for empty sequence.
        alphas = self.transitions[self.START_TAG_ID] + emissions[:, 1]

        # Here we iterate by sequence length
        # However, iterate by batch index is more readable in <<统计学习方法>>.
        # HACK: we start from index 2 because we use bert, if we use XLnet or others.
        # rewrite the progress.
        for seq_idx in range(2, seq_length):
            # (bs, num_labels) -> (bs, 1, num_labels)
            e_scores = emissions[:, seq_idx].unsqueeze(1)

            # (num_labels, num_labels) -> (1, num_labels, num_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, num_labels) -> (bs, num_labels, 1)
            a_scores = alphas.unsqueeze(2)

            # Set alphas if the mask is valid, otherwise keep the current values
            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim=1)

            is_valid = mask[:, seq_idx].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        # Add the scores for the final transition
        last_transition = self.transitions[:, self.END_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a *log* of sums of exp
        return torch.logsumexp(end_scores, dim=1)

    def decode(self, emissions, mask):
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions, mask):
        """Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.

        Args:
            emissions (torch.Tensor): (batch_size, seq_len, num_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)

        Returns:
            best_sequences: list of lists of int: the best viterbi sequence of labels for each batch
            max_final_scores: torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length, _ = emissions.shape
        last_valid_idx = mask.int().sum(1) - 1
        zero = torch.zeros(1).long().to(mask.device)
        # For bert whose tokens like ['[CLS]', xx, '[SEP]']
        mask = mask.index_put((torch.arange(batch_size), last_valid_idx), zero)

        # In the first iteration, BOS will have all the scores and then, the max
        # NOTE: the alphas here is totoally different from the alphas in `_compute_log_partition`
        alphas = self.transitions[self.START_TAG_ID].unsqueeze(0) \
                + emissions[:, 1]

        backpointers = []

        for i in range(2, seq_length):
            # (bs, num_labels) -> (bs, 1, num_labels)
            e_scores = emissions[:, i].unsqueeze(1)

            # (num_labels, num_labels) -> (1, num_labels, num_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, num_labels)  -> (bs, num_labels, 1)
            a_scores = alphas.unsqueeze(2)

            # combine current scores with previous alphas
            scores = e_scores + t_scores + a_scores

            # so far is exactly like the forward algorithm,
            # but now, instead of calculating the logsumexp,
            # we will find the highest score and the tag associated with it
            max_scores, max_score_tags = torch.max(scores, dim=1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas

            # add the max_score_tags for our list of backpointers
            # max_scores has shape (batch_size, num_labels) so we transpose it to
            # be compatible with our previous loopy version of viterbi
            backpointers.append(max_score_tags.t())

        # Add the scores for the final transition
        last_transition = self.transitions[:, self.END_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # Get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # Find the best sequence of labels for each sample in the batch
        best_sequences = []
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_final_idx = last_valid_idx[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            # NOTE: we use sample_final_idx -2 because `backpointers`
            # only record len_sentences - 3 pointers.
            sample_backpointers = backpointers[:sample_final_idx - 2]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag,
                                               sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return best_sequences, max_final_scores

    def _find_best_path(self, batch_idx, best_tag, backpointers):
        """Auxiliary function to find the best path sequence for a specific sample.

            Args:
                batch_idx (int): sample index in the range [0, batch_size)
                best_tag (int): tag which maximizes the final score
                backpointers (list of lists of tensors): list of pointers with
                shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
                represents the length of the ith sample in the batch

            Returns:
                list of int: a list of tag indexes representing the bast path
        """

        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][batch_idx].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path
