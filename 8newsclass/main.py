from os import listdir
from os.path import isfile, join
import csv


def get_paths(path, categs):
    data = []
    idx = 0
    for categ in categs:
        data.append([])
        dir_path = join(path, categ)
        for doc in listdir(dir_path):
            file_path = join(dir_path, doc)
            if (isfile(file_path)):
                data[idx].append(file_path)
        ++idx
    return data


def get_rawdocuments(path, categs):
    data = []
    idx = 0
    for categ in categs:
        data.append([])
        dir_path = join(path, categ)
        for doc in listdir(dir_path):
            file_path = join(dir_path, doc)
            if (isfile(file_path)):
                data[idx].append(file_path)
        ++idx
    return data


def get_featuremaps(paths_by_class, feature_extractor):
    fmaps = []
    for paths in paths_by_class:
        fmaps.append([])
        for path in paths:
            fmaps[-1].append(feature_extractor(open(path).readlines()))
    return fmaps


def to_dataset(labels, featuremaps):
    cols = len(labels) + 1
    rows = 0
    for featuremaps_by_class in featuremaps:
        rows += len(featuremaps_by_class)

    dataset = [None] * rows
    idx = 0
    c1 = 0
    for featuremaps_by_class in featuremaps:
        ++idx
        print("CLASS: " + str(idx) + " len: " + str(len(featuremaps_by_class)))
        dataset
        for featuremap in featuremaps_by_class:
            dataset[c1] = [0] * cols
            c1 = c1 + 1
            idx2 = 0
            print(c1)
            """keys = featuremap.keys()
            for label in labels:
                if label in keys:
                    dataset[-1][idx2] = featuremap[label]
                else:
                    dataset[-1][idx2] = 0
                dataset[-1][idx2] = 7
                ++idx2
            dataset[-1].append(idx)
            """
    return dataset


def write_dataset(dataset, filename):
    file = open(filename, "wb")
    writer = csv.writer(file, delimiter=' ')
    writer.writerows(dataset)


class CountTerms:
    termset = set()

    def extract_tfmap(self, content):
        tfmap = {}
        content = " ".join(content)
        terms = content.split()
        for term in terms:
            self.termset.add(term)
            if (term in tfmap.keys()):
                tfmap[term] = tfmap[term] + 1
            else:
                tfmap[term] = 1
        return tfmap


def idf(test):
    return None


def zero_one(input):
    return None

#def test():
"""
import main
categs = ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space"]
paths = main.get_paths("/Users/mac/PycharmProjects/mlcodes/classification/8news/ex6-20news", categs)
ct = main.CountTerms()
tfmaps = main.get_featuremaps(paths, ct.extract_tfmap);
tfdataset = main.to_dataset(ct.termset, tfmaps)
main.write_dataset("ex6.data", tfdataset)
    #zods = get_features(paths, zero_one)
"""
