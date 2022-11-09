from ClassesDB.DatabaseIDs import *

class Tracker:

    def __init__(self):

        # Database of identification information
        self.identifies_db = None

        # Database of center locations
        self.center = []

    def _getCenter(self, person):
        x = (person[0] + person[2]) / 2
        y = (person[1] + person[3]) / 2
        return (x, y)

    def _getDistance(self, person, index):

        (x1, y1) = self.center[index]
        (x2, y2) = self._getCenter(person)
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        u = b - a
        return np.linalg.norm(u)

    def _isOverlap(self, persons, index):
        [x1, y1, x2, y2] = persons[index]
        for i, person in enumerate(persons):
            if (index == i):
                continue
            if (max(person[0], x1) <= min(person[2], x2) and max(person[1], y1) <= min(person[3], y2)):
                return True
        return False

    # Cosine similarity
    # reference document: https://github.com/kodamap/person_reidentification
    def _cos_similarity(self, X, Y):

        m = X.shape[0]
        Y = Y.T
        return np.dot(X, Y) / (np.linalg.norm(X.T, axis=0).reshape(m, 1) * np.linalg.norm(Y, axis=0))

    def getIds(self, identifies, persons):

        if (identifies.size == 0):
            return []
        if self.identifies_db is None:
            self.identifies_db = identifies
            for person in persons:
                self.center.append(self._getCenter(person))

        similaritys = self._cos_similarity(identifies, self.identifies_db)
        similaritys[np.isnan(similaritys)] = 0
        ids = np.nanargmax(similaritys, axis=1)

        for i, similarity in enumerate(similaritys):
            persionId = ids[i]
            d = self._getDistance(persons[i], persionId)
            # print("persionId:{} {} distance:{}".format(persionId, similarity[persionId], d))
            # If similarity is greater than 0.95 and there is no overlap, the identification information is updated
            if (similarity[persionId] > 0.95):
                if (self._isOverlap(persons, i) == False):
                    self.identifies_db[persionId] = identifies[i]

            # If similarity is less than 0.5 and the distance is far, register a new ID
            elif (similarity[persionId] < 0.5):
                if (d > 500):
                    print("distance:{} similarity:{}".format(d, similarity[persionId]))
                    self.identifies_db = np.vstack((self.identifies_db, identifies[i]))
                    self.center.append(self._getCenter(persons[i]))
                    ids[i] = len(self.identifies_db) - 1
                    print("> append DB size:{}".format(len(self.identifies_db)))

        # If there are duplicate IDs, the one with the lower confidence level is invalidated.
        for i, a in enumerate(ids):
            for e, b in enumerate(ids):
                if (e == i):
                    continue
                if (a == b):
                    if (similarity[a] > similarity[b]):
                        ids[i] = -1
                    else:
                        ids[e] = -1

        return ids.tolist()
