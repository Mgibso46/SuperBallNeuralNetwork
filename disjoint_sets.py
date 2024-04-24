class DisjointSets:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            return None  # Element not found
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x is None or root_y is None:
            return False  # One of the elements doesn't exist
        if root_x == root_y:
            return False  # Elements are already in the same set

        # Merge smaller set into larger set to maintain balanced tree
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size.get(root_y, 1)  # Add size of root_y to root_x
        return True  # Union successful

    def get_set_size(self, x):
        root = self.find(x)
        if root is None:
            return 0  # Element not found
        return self.size.get(root, 0)
