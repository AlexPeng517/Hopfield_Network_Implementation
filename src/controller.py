from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
from hopfield_UI import Ui_MainWindow
from PyQt5 import QtWidgets
from dataloader import Dataloader
from hopfield import Hopfield


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.output_pixmap = None
        self.input_pixmap = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.train_filename = ""
        self.test_filename = ""
        self.hopfield = None
        self.dataloader = None
    def setup_control(self):
        self.ui.openFileButton.clicked.connect(self.open_train_file)
        self.ui.openFileButton_2.clicked.connect(self.open_test_file)
        self.ui.pushButton.clicked.connect(self.train)
        self.ui.pushButton_2.clicked.connect(self.test)

    def open_train_file(self):
        self.train_filename, filetype = QFileDialog.getOpenFileNames(self, "Open file", "./", "*.txt")
        self.ui.selectedFileURLTextEdit.setText(str(self.train_filename))
        print("filename", self.train_filename)
        if len(self.train_filename) == 0:
            self.ui.selectedFileURLTextEdit.setText("Failed! Please Try Again!")
            return

    def open_test_file(self):
        self.test_filename, filetype = QFileDialog.getOpenFileNames(self, "Open file", "./", "*.txt")
        self.ui.selectedFileURLTextEdit_2.setText(str(self.test_filename))
        print("filename", self.test_filename)
        if len(self.test_filename) == 0:
            self.ui.selectedFileURLTextEdit_2.setText("Failed! Please Try Again!")
            return

    def train(self):
        input_dim1 = self.ui.input_dim1.value()
        input_dim2 = self.ui.input_dim2.value()
        self.hopfield = Hopfield(self.train_filename,input_dim1,input_dim2)
        self.hopfield.train_hopfield()
        self.dataloader = Dataloader(self.train_filename,self.test_filename,input_dim1,input_dim2)
        range_message = "0 to "+str(self.dataloader.train_entries_num-1)
        self.ui.label_range.setText(range_message)


    def test(self):
        is_noisy = self.ui.checkBox.isChecked()
        is_threshold = self.ui.checkBox_2.isChecked()
        is_async = self.ui.checkBox_3.isChecked()
        chosen_case_index = self.ui.spinBox.value()
        noise_degree = self.ui.DegreeDoubleSpinBox.value()
        if is_noisy:
            test_case = self.dataloader.gen_noisy_from_clean(chosen_case_index,noise_degree)
        else:
            test_case = self.dataloader.get_test_sample(chosen_case_index)
        self.hopfield.test(test_case,enable_threshold=is_threshold,enable_async=is_async)
        self.input_pixmap = QPixmap('0.png')
        self.output_pixmap = QPixmap('1.png')
        self.ui.input_image.setPixmap(self.input_pixmap)
        self.ui.output_image.setPixmap(self.output_pixmap)
        self.ui.textEdit.clear()
        self.ui.textEdit.setText(str(self.hopfield.energy_history))







if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())