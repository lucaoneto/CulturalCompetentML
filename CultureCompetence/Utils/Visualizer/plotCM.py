from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

cm = [
    [74.5, 124.7, 5.5],
    [0.3, 199.6, 4.8],
    [4.1, 12.3, 169.2]
]
cm = np.asarray(cm)/ np.sum(np.sum(cm))
classes =  ['Chin.', 'Fren.', 'Turk.'] # ['Indian', 'Japan.', 'Scand.']

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes)
disp.plot()
plt.show()