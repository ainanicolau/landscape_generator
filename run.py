from PyQt5 import QtWidgets
import sys

from gui import CreateLandscapeGUI

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CreateLandscapeGUI()
    window.show()
    sys.exit(app.exec_())
