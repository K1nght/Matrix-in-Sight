import numpy as np
import matplotlib.pyplot as plt


def distance(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


class Node(object):
    def __init__(self, i):
        self.index = i
        self.left = None
        self.right = None
        self.up = None
        self.down = None


class Graph(object):
    def __init__(self, matrix, max_l, mnist=False):
        self.__row = 0
        self.__col = 0
        self.graph = []
        self.max_l = max_l
        self.shape = (self.__row, self.__col)
        self.__mnist = mnist

        for i in range(len(matrix)):
            self.graph.append(Node(i))
        for i in range(len(matrix)):
            id1, x1, y1, xmin1, ymin1, xmax1, ymax1 = matrix[i]
            for n in self.graph:
                if n.index == i:
                    node = n

            lmd, rmd, umd, dmd = float('inf'), float('inf'), float('inf'), float('inf')
            l, r, u, d = None, None, None, None
            for other in self.graph:
                j = other.index
                if i == j:
                    continue
                id2, x2, y2, xmin2, ymin2, xmax2, ymax2 = matrix[j]
                # left and right
                if (abs(y1-y2) <= self.max_l/2 and mnist) or (ymin2 < y1 < ymax2 and ymin1 < y2 < ymax1):
                    #      *
                    #      |
                    #   *--j--i
                    #      |
                    #      *
                    if x1 > x2:
                        if distance(x1, y1, x2, y2) < lmd:
                            lmd = distance(x1, y1, x2, y2)
                            l = other
                    #       *
                    #       |
                    #    i--j--*
                    #       |
                    #       *
                    else:
                        if distance(x1, y1, x2, y2) < rmd:
                            rmd = distance(x1, y1, x2, y2)
                            r = other

                # up and down
                if (abs(x2-x1) <= self.max_l/2 and mnist) or (xmin2 < x1 < xmax2 and xmin1 < x2 < xmax1):
                    #      i
                    #      |
                    #   *--j--*
                    #      |
                    #      *
                    if y1 > y2:
                        if distance(x1, y1, x2, y2) < dmd:
                            dmd = distance(x1, y1, x2, y2)
                            d = other
                    #       *
                    #       |
                    #    *--j--*
                    #       |
                    #       i
                    else:
                        if distance(x1, y1, x2, y2) < umd:
                            umd = distance(x1, y1, x2, y2)
                            u = other
            node.left, node.right, node.up, node.down = l, r, u, d

        self.start = None
        for node in self.graph:
            if node.left is None and node.down is None:
                self.start = node
        p = self.start
        while p:
            self.__col += 1
            p = p.right
        p = self.start
        while p:
            self.__row += 1
            p = p.up
        self.shape = (self.__row, self.__col)

    def print_graph(self, matrix):
        for node in self.graph:
            i = node.index
            print(matrix[i], "left[{}],right[{}],up[{}],down[{}]".format(
                matrix[node.left.index][0] if node.left else None,
                matrix[node.right.index][0] if node.right else None,
                matrix[node.up.index][0] if node.up else None,
                matrix[node.down.index][0] if node.down else None,
            ))

    def plot_graph(self, matrix):
        plt.grid(True)
        for i, x, y, xmin, ymin, xmax, ymax in matrix:
            plt.plot(x, y, 'o')
            print('(%d,%d,%d)' % (i, x, y))
            plt.annotate('(%d,%d,%d)' % (i, x, y), xy=(x, y), xytext=(-20, 10), textcoords='offset points')
        plt.show()

    def get_matrix(self, matrix):
        Matrix1 = [[0] * self.__col for _ in range(self.__row)]
        p = self.start
        i, j = 0, 0
        while p:
            q = p
            while q:
                # print(i, j, matrix[q.index])
                Matrix1[i][j] = matrix[q.index][0]
                q = q.right
                j += 1
            p = p.up
            i += 1
            j = 0
        Matrix1 = np.array(Matrix1)
        return Matrix1


class calculator(object):
    def __init__(self, classes):
        self.operator = None
        self.matrixs = []
        self.points = {}
        self.T = []
        self.max_l = 0
        for label in classes:
            self.points[label] = []

    def get_from_points(self, points, max_l):
        self.points = points
        self.max_l = max_l

    def get_from_txt(self, txtPath):
        txtFile = open(txtPath)
        txtList = txtFile.readlines()
        for oneline in txtList:
            idx, index, label, xmin, ymin, xmax, ymax, score = oneline.split(" ")
            xmin, ymin, xmax, ymax, score = int(xmin), int(ymin), int(xmax), int(ymax), float(score[:-1])
            self.max_l = max(self.max_l, xmax-xmin, ymax-ymin)
            x, y = (xmin + xmax) / 2, (ymin + ymax) / 2

            # 防止数字重叠
            if label == 'number':
                drop = False
                for i, (idx2, x2, y2, xmin2, ymin2, xmax2, ymax2, score2) in enumerate(self.points['number']):
                    if (xmin2 < x < xmax2 and ymin2 < y < ymax2) or (xmin < x2 < xmax and ymin < y2 < ymax):
                        if (xmax - xmin) * (ymax - ymin) < (xmax2 - xmin2) * (ymax2 - ymin2):
                            drop = True
                            break
                        else:
                            self.points['number'][i] = [int(idx), x, y, xmin, ymin, xmax, ymax, score]
                            drop = True
                            break
                if not drop:
                    self.points[label].append([int(idx), x, y, xmin, ymin, xmax, ymax, score])
            # 其他情况取分高的
            else:
                drop = False
                for l in self.points.keys():
                    for i, (idx2, x2, y2, xmin2, ymin2, xmax2, ymax2, score2) in enumerate(self.points[l]):
                        if (xmin2 < x < xmax2 and ymin2 < y < ymax2) or (xmin < x2 < xmax and ymin < y2 < ymax):
                            if score < score2:
                                drop = True
                                break
                            else:
                                del self.points[l][i]
                if not drop:
                    self.points[label].append([int(idx), x, y, xmin, ymin, xmax, ymax, score])
        print(self.points)

    def get_operator(self):
        for l in self.points.keys():
            if l == "add" or l == "multi" or l == "minus":
                if len(self.points[l]) == 1 and self.operator is None:
                    self.operator = l
                elif len(self.points[l]) > 1 or (len(self.points[l]) == 1 and self.operator is not None):
                    raise NotImplementedError("There are more than one operator in the picture!")
        # 图中只要一个矩阵时默认为求行列式
        if self.operator is None:
            self.operator = "det"
        print("The operator is %s" % self.operator)

    def sort_matrix(self):
        if self.operator == "det":
            matrix = []
            for i, (id, x, y, xmin, ymin, xmax, ymax, _) in enumerate(self.points['number']):
                matrix.append([id, x, y, xmin, ymin, xmax, ymax])
            print("The len of matrix is %d" % len(matrix))

            # check_dim = True
            # for i in range(10):
            #     if len(matrix) == i ** 2:
            #         check_dim = False
            # if check_dim:
            #     raise NotImplementedError(
            #         "It's not a square, so cannot do the det operation!"
            #     )

            self.matrixs.append(matrix)
            if len(self.points["T"]) == 1:
                self.T.append(True)
            elif len(self.points["T"]) == 0:
                self.T.append(False)
            else:
                raise NotImplementedError(
                    "The number of the transpose operator is not match with the number of the matrix!"
                )
        else:
            _, operator_x, operator_y, _, _, _, _, _ = self.points.get(self.operator, None)[0]
            matrix1 = []
            matrix2 = []
            for i, (id, x, y, xmin, ymin, xmax, ymax, _) in enumerate(self.points['number']):
                if x < operator_x:
                    matrix1.append([id, x, y, xmin, ymin, xmax, ymax])
                else:
                    matrix2.append([id, x, y, xmin, ymin, xmax, ymax])
            print("The number of the elements in left matrix is %d" % len(matrix1))
            print("The number of the elements in right matrix is %d" % len(matrix2))
            self.matrixs = [matrix1, matrix2]

            if self.operator == "add" or self.operator == "minus":
                if len(matrix1) != len(matrix2):
                    raise NotImplementedError(
                        "The len of left matrix is not the same as the len of right matrix,",
                        " so they cannot do the %s operation" % self.operator
                    )

            self.T = [False, False]
            if len(self.points["T"]) > 2:
                raise NotImplementedError(
                    "The number of the transpose operator is not match with the number of the matrix!"
                )
            else:
                for i, (_, x, y, xmin, ymin, xmax, ymax, _) in enumerate(self.points["T"]):
                    if x < operator_x:
                        if self.T[0]:
                            raise NotImplementedError(
                                "The number of the transpose operator is not match with the number of the matrix!"
                            )
                        else:
                            self.T[0] = True
                    else:
                        if self.T[1]:
                            raise NotImplementedError(
                                "The number of the transpose operator is not match with the number of the matrix!"
                            )
                        else:
                            self.T[1] = True

    def __call__(self):
        self.get_operator()
        self.sort_matrix()
        if self.operator == "det":
            graph = Graph(self.matrixs[0], self.max_l)
            matrix = graph.get_matrix(self.matrixs[0])
            # print(self.matrixs[0])
            print("The shape of the matrix is ", graph.shape)
            # graph.plot_graph(matrix)
            if self.T[0]:
                matrix = matrix.T
            print(matrix)
            return int(np.linalg.det(matrix))
        else:
            # print(self.matrixs[0], self.max_l)
            graph1 = Graph(self.matrixs[0], self.max_l)
            graph2 = Graph(self.matrixs[1], self.max_l)
            matrix1 = graph1.get_matrix(self.matrixs[0])
            if self.T[0]:
                matrix1 = matrix1.T
            matrix2 = graph2.get_matrix(self.matrixs[1])
            if self.T[1]:
                matrix2 = matrix2.T
            print("The shape of the matrix1 is", graph1.shape)
            print("The shape of the matrix2 is", graph2.shape)
            matrix1 = np.array(matrix1)
            matrix2 = np.array(matrix2)
            print(matrix1)
            if self.operator == "add":
                print("+")
            elif self.operator == "minus":
                print("-")
            elif self.operator == "mulit":
                print("x")
            print(matrix2)
            print("=")
            if self.operator == "add":
                return matrix1 + matrix2
            elif self.operator == "minus":
                return matrix1 - matrix2
            elif self.operator == "multi":
                return np.dot(matrix1, matrix2)

