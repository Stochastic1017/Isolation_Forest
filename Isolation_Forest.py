
import numpy as np

class  IsolationForestAnomalyDetector():

    def __init__(self, X):
        self.X = X
        self.M, self.d = self.X.shape

    def binary_partition(self, S=None):

        # fallback to use all data if X is None
        if S is None:
            S = self.X
            
        # base case: X is a single isolated point
        if len(S) <= 1:
            return (S, np.array([]).reshape(0, self.d), 0, 0)
        
        # Choose a random axis from {0, ..., d-1} on which to cut
        # Discrete uniform distribution, i.e., P(X=i) = 1/d for all i
        q = np.random.randint(low=0, high=self.d)

        # Access the chosen axis (column) chosen above
        S_q = S[:, q]

        # Choose a random pointon chosen axis (column) from the respective (min, max)
        # Continuous uniform distribution, i.e., P(a <= X <= b) = (b-a)/(max-min)
        p = np.random.uniform(low=S_q.min(), high=S_q.max())

        # Apply condition to split points (rows of S) from left and right of random cut
        split_condition = S_q < p
        S_left = S[np.where(split_condition)[0]]   # Split S where condition is TRUE (i.e., left)
        S_right = S[np.where(~split_condition)[0]] # Split S where condition is FALSE (i.e., right)

        return (S_left, S_right, q, p)
        
    def iTree(self, X=None, counter=0, limit=100):

        # fallback to use all data if X is None
        if X is None:
            X = self.X

        # base case: all points are isolated or height limit reached
        if (len(X) <= 1) or (counter >= limit) or (np.all(X == X[0])):
            # Create external node when point is isolated
            return {'type': 'external', 'size': len(X)}
        
        # Split left and right from random point on random axis
        left_split, right_split, split_axis, split_point = self.binary_partition(X)

        # Create internal node with recursive calls
        return {
            'type': 'internal',
            'left': self.iTree(left_split, counter+1, limit),  # further split left into two, add 1 to the counter
            'right': self.iTree(right_split, counter+1,limit), # further split right into two, add 1 to the counter
            'split_axis': split_axis,
            'split_point': split_point
        }

    def expected_length_of_unsuccessful_search_in_RBST(self, length):

        # trivial case
        if length <= 1:
            return 0.0

        # base case: tree of size 2, return length 1.0
        if length == 2:
            return 1.0

        # Approximation of harmonic series 1 + 1/2 + 1/3 + ... + 1/(length-1)
        H = np.log(length-1) + np.euler_gamma

        # RBST expected path in unsuccessful search formula
        # Refer: The Art of Computer Programming (Volume 3) - Donald Knuth
        return 2 * (H - (1 - 1/length))

    def path_length_by_point(self, point, iTree, current_length=0):

        # base case: if node is external, compute final path length
        if iTree['type'] == 'external':
            return current_length + self.expected_length_of_unsuccessful_search_in_RBST(iTree['size'])

        else:
            # Fetch the axis and point of split for this internal node
            split_axis = iTree['split_axis']
            split_point = iTree['split_point']

            # Go left if point is on the left-side of the split
            if point[split_axis] < split_point:
                return self.path_length_by_point(point, iTree['left'], current_length+1)

            # Go right if point is on the right-side of the split
            else:
                return self.path_length_by_point(point, iTree['right'], current_length+1)

    def iForest(self):

        # Initialize array to store all iTrees
        self.forest = []
        for idx in range(self.M):
            # bootstrap sample full dataset
            # max_samples=1.0 => 1:1 ratio of data and samples (with replacement)
            bootstrapped_X = self.X[
                np.random.choice(self.M, size=self.M, replace=True)
            ]
            # maximum depth = log2(n)
            limit = int(np.ceil(np.log2(len(bootstrapped_X))))
            iTree_idx = self.iTree(bootstrapped_X, counter=0, limit=limit)
            self.forest.append(iTree_idx)

        return self.forest

    def anomaly_scores(self):

        if not hasattr(self, 'forest'):
            # build an iForest if not already built
            self.iForest()

        # Average path length of unsuccessful search in RBST
        C_M = self.expected_length_of_unsuccessful_search_in_RBST(self.M)
        scores = []
        for point in self.X:
            # Average path length for each point across all trees
            E_hx = np.mean([self.path_length_by_point(point, itree, 0) for itree in self.forest])
            # Anomaly scores in [0,1], higher -> more anomalous
            scores.append( 2**(-E_hx / C_M) )

        return scores
