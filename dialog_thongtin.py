from PyQt6 import QtCore, QtGui, QtWidgets


class Dialog_ThongTin(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(600, 500)
        Dialog.setStyleSheet("background-color: rgb(228,228,228);\n"
"color: rgb(0, 0, 0);")
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(parent=Dialog)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(218, 218, 218);")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "- Phần mềm xem camera giao thông và giám sát kiểm tra các lỗi \n"
" vi phạm giao thông cơ bản. Ý tưởng phần mềm là mô phỏng lại công\nnghệ phát hiện lỗi vi phạm giao thông của Sở giao "
                                                "thông vận tải"
"\n"
"\n"
"- Nguồn dữ liệu :\n"
"\n"
"+ TTGT TP . Hồ Chí Minh\n"
"+ TTGT Bình Định\n"
"+ App Huế S\n"
"+ App iHanoi\n"
"+ App Bạc Liêu Smart\n"
"+ App Bình Dương Số\n"
"\n"
"- Phần mềm được phát triển bởi :\n"
"\n"
" Huỳnh Mai Nhật Minh\n"
" Lê Huy Hoàng\n"
" Trần Hải Bằng\n"
" Trần Minh Khang "))
