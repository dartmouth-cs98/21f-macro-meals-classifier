class ClassificationMap:

    index2classification = [
        "apple",
        "banana",
        "beans",
        "carrot",
        "cheese",
        "cucumber",
        "onion",
        "orange",
        "pasta",
        "pepper",
        "qiwi",
        "sauce",
        "tomato",
        "watermelon"
    ]

    def __init__(self):
        pass

    def get_classification(self, i):
        return self.index2classification[int(i) - 1]
