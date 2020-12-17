#DEPENDENCIES
import sys
import platform
import time
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent, QThread, QThreadPool, pyqtSlot, pyqtSignal)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
from queue import Queue
from datetime import datetime

# GUI FILE
from ui_fd import Ui_MainWindow
import Fault_Detection_Arzy
import Subscriber

#Default plot colors
mpl.rcParams['axes.facecolor'] ='3c3c3c'
mpl.rcParams["figure.facecolor"] = '3c3c3c'
mpl.rcParams['axes.edgecolor'] = 'ffffff'
mpl.rcParams['figure.edgecolor'] = 'ffffff'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['xtick.color'] ='ffffff'
mpl.rcParams['ytick.color'] = 'ffffff'

class MainWindow(QMainWindow):
    # Plot variables
    normal_reference = 0
    X = 0
    pred = 0
    # Plot Signal
    got_normal_reference = pyqtSignal()
    got_X = pyqtSignal()
    got_pred = pyqtSignal()

    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowIcon(QIcon("icon/machine-learning.png"))
        self.setWindowTitle("Fault Detection - UGM")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 120))
        self.ui.centralwidget.setGraphicsEffect(self.shadow)
        self.ui.SVMButton.setCheckable(True)
        self.ui.FDAButton.setCheckable(True)

        # Set page to Status Page
        self.ui.Pages.setCurrentWidget(self.ui.status)

        # Setting up canvases
        ## figure instances to plot on 
        self.predfigure = plt.figure()
        self.Xfigure = plt.figure()
        self.SVMfigure = plt.figure()
        ## Canvas widget needs figure as parameter
        self.predcanvas = FigureCanvas(self.predfigure)
        self.Xcanvas = FigureCanvas(self.Xfigure)
        self.SVMcanvas = FigureCanvas(self.SVMfigure)
        # adding canvas to the layout 
        self.ui.predLayout.addWidget(self.predcanvas)
        self.ui.XLayout.addWidget(self.Xcanvas)
        self.ui.SVMLayout.addWidget(self.SVMcanvas)

        ## Show Window
        self.show()

        #DRAG AND MOVE WINDOW
        def moveWindow(event):
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()
        self.ui.topBar.mouseMoveEvent = moveWindow

        #MINIMIZE AND CLOSE WINDOW
        self.ui.btn_minimize.clicked.connect(lambda: self.showMinimized())
        self.ui.btn_close.clicked.connect(lambda: self.close())

        # connecting signal and slot
        ## Switch pages if button pressed
        self.ui.statusButton.clicked.connect(lambda: self.ui.Pages.setCurrentWidget(self.ui.status))
        self.ui.statusButton.clicked.connect(lambda: self.ui.pageTitle.setText("System Status Screen |"))
        self.ui.mqttButton.clicked.connect(lambda: self.ui.Pages.setCurrentWidget(self.ui.mqtt))
        self.ui.mqttButton.clicked.connect(lambda: self.ui.pageTitle.setText("Connection Screen |"))
        self.ui.controlButton.clicked.connect(lambda: self.ui.Pages.setCurrentWidget(self.ui.control))
        self.ui.controlButton.clicked.connect(lambda: self.ui.pageTitle.setText("Settings Screen |"))
        self.ui.resultButton.clicked.connect(lambda: self.ui.Pages.setCurrentWidget(self.ui.result))
        self.ui.resultButton.clicked.connect(lambda: self.ui.pageTitle.setText("Results Screen |"))
        ## Connect to mqtt connect if button pressed
        self.ui.connectButton.clicked.connect(lambda: self.mqtt_connect())
        ## Connect to GUI dispatcher if new message received
        self.client = Subscriber.MyMQTTClass(self)
        self.client.messageSignal.connect(self.GUI_dispatcher)
        ## Connect to plot functions if plot variables changed
        self.got_normal_reference.connect(self.drawX)
        self.got_pred.connect(self.drawPred)
        self.got_X.connect(self.drawSVM)
        self.got_X.connect(self.drawX)
        #get fault detection settings and send to fault_detection
        ## Get and send default control settings
        self.reference = self.ui.referenceLine.text()
        Fault_Detection_Arzy.faultDetection.getReference(int(self.reference))
        self.buffer = self.ui.bufferLine.text()
        Fault_Detection_Arzy.faultDetection.getBuffer(int(self.buffer))
        self.kernel = str(self.ui.kernelBox.currentText())
        Fault_Detection_Arzy.faultDetection.getKernel(self.kernel)
        self.gamma = self.ui.gammaLine.text()
        Fault_Detection_Arzy.faultDetection.getGamma(float(self.gamma))
        self.degree = self.ui.degreeLine.text()
        Fault_Detection_Arzy.faultDetection.getDegree(int(self.degree))
        self.eig = self.ui.eigLine.text()
        Fault_Detection_Arzy.faultDetection.getEig(int(self.eig))
        self.wcssThres = self.ui.wcssLine.text()
        Fault_Detection_Arzy.faultDetection.getWcssThres(int(self.wcssThres))
        self.classifier = "Support Vector Machines"
        self.ui.SVMButton.setChecked(True)
        Fault_Detection_Arzy.faultDetection.getClassifier(self.classifier)

        ## Get and send new control settings
        def getReference():
            self.canConnect1 = True
            self.reference = self.ui.referenceLine.text()
            if self.reference:
                try:
                    self.reference = int(self.reference)
                except ValueError:
                    self.canConnect1 = False
            else:
                self.reference = 100
            Fault_Detection_Arzy.faultDetection.getReference(self.reference)
            self.client.getRefs(self.reference)
        def getBuffer():
            self.canConnect2 = True
            self.buffer = self.ui.bufferLine.text()
            if self.buffer:
                try:
                    self.buffer = int(self.buffer)
                except ValueError:
                    self.canConnect2 = False
            else:
                self.buffer = 50
            Fault_Detection_Arzy.faultDetection.getBuffer(self.buffer)
            self.client.getBuffs(self.buffer)
        def getKernel():
            self.kernel = str(self.ui.kernelBox.currentText())
            Fault_Detection_Arzy.faultDetection.getKernel(self.kernel)
        def getGamma():
            self.canConnect3 = True
            self.gamma = self.ui.gammaLine.text()
            if self.gamma:
                try:
                    self.gamma = float(self.gamma)
                except ValueError:
                    self.canConnect3 = False
            else:
                self.gamma = 0.0085
            Fault_Detection_Arzy.faultDetection.getGamma(self.gamma)
        def getDegree():
            self.canConnect4 = True
            self.degree = self.ui.degreeLine.text()
            if self.degree:
                try:
                    self.degree = int(self.degree)
                except ValueError:
                    self.canConnect4 = False
            else:
                self.degree = 0
            Fault_Detection_Arzy.faultDetection.getDegree(self.degree)
        def getEig():
            self.canConnect5 = True
            self.eig = self.ui.eigLine.text()
            if self.eig:
                try:
                    self.eig = int(self.eig)
                except ValueError:
                    self.canConnect5: False
            else:
                self.eig = 589
            Fault_Detection_Arzy.faultDetection.getEig(self.eig)
        def getWcssThres():
            self.canConnect6 = True
            self.wcssThres = self.ui.wcssLine.text()
            if self.wcssThres:
                try:
                    self.wcssThres = float(self.wcssThres)
                except ValueError:
                    self.canConnect6: False
            else:
                self.wcssThres = 110
            Fault_Detection_Arzy.faultDetection.getWcssThres(self.wcssThres)
        def getClassifierSVM():
            self.classifier = self.ui.SVMButton.text()
            self.ui.SVMButton.setChecked(True)
            self.ui.FDAButton.setChecked(False)
            Fault_Detection_Arzy.faultDetection.getClassifier(self.classifier)

        def getClassifierFDA():
            self.classifier = self.ui.FDAButton.text()
            self.ui.FDAButton.setChecked(True)
            self.ui.SVMButton.setChecked(False)
            Fault_Detection_Arzy.faultDetection.getClassifier(self.classifier)


        #get and send default connection settings
        self.ip = self.ui.ipLine.text()
        self.client.getIp(self.ip)
        self.port = self.ui.portLine.text()
        self.client.getPort(self.port)
        self.feature = self.ui.featureLine.text()
        self.client.getTopics(self.feature)
        
        #get and send new connection settings
        def getIp():
            self.ip = self.ui.ipLine.text()
            self.client.getIp(self.ip)

        def getPort():
            self.port = self.ui.portLine.text()
            self.client.getPort(self.port)

        def getTopics():
            self.feature = self.ui.featureLine.text()
            self.client.getTopics(self.feature)

        #Connect to get functions if entry changed
        self.ui.referenceLine.textChanged.connect(getReference)
        self.ui.bufferLine.textChanged.connect(getBuffer)
        self.ui.kernelBox.currentIndexChanged.connect(getKernel)
        self.ui.gammaLine.textChanged.connect(getGamma)
        self.ui.degreeLine.textChanged.connect(getDegree)
        self.ui.eigLine.textChanged.connect(getEig)
        self.ui.wcssLine.textChanged.connect(getWcssThres)
        self.ui.SVMButton.clicked.connect(getClassifierSVM)
        self.ui.FDAButton.clicked.connect(getClassifierFDA)
        self.ui.ipLine.textChanged.connect(getIp)
        self.ui.portLine.textChanged.connect(getPort)
        self.ui.featureLine.textChanged.connect(getTopics)

        #input message into Table
        self.client.on_messageSignal.connect(self.inputTable)

    def addTableRow(self, row, current_row):
        current_column = 0
        for item in row:
            cell = QTableWidgetItem((item).decode('utf-8'))
            self.ui.mqttTable.setItem(current_row, current_column, cell)
            current_column +=1

    @pyqtSlot(list)
    def inputTable(self, buffer_pocket):
        data = buffer_pocket[0]
        rows = data.values.tolist()
        current_row = 0
        for row in rows:
            self.addTableRow(row, current_row)
            current_row +=1

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    mqtt_connected = False #Default value for mqtt_connected
    def mqtt_connect(self):
        if self.mqtt_connected == False:
            self.ui.statusStat.setText("Connection Established")
            self.ui.connectButton.setText("Disconnect")
            self.ui.connectButton.setGeometry(80, 68, 81, 21)
            self.ui.connectButton.setStyleSheet("color:rgb(203, 105, 105)")
            self.ui.mqttTable.setRowCount(int(self.buffer))
            self.ui.mqttTable.setColumnCount(int(self.feature))
            self.ui.ipLine.setDisabled(True)
            self.ui.portLine.setDisabled(True)
            self.ui.featureLine.setDisabled(True)
            self.ui.referenceLine.setDisabled(True)
            self.ui.bufferLine.setDisabled(True)
            self.ui.kernelBox.setDisabled(True)
            self.ui.gammaLine.setDisabled(True)
            self.ui.degreeLine.setDisabled(True)
            self.ui.eigLine.setDisabled(True)
            self.ui.wcssLine.setDisabled(True)
            self.mqtt_connected = True
            self.client.start()
        elif self.mqtt_connected == True:
            self.ui.statusStat.setText("No Connection Established")
            self.ui.connectButton.setText("Connect")
            self.ui.connectButton.setGeometry(160, 68, 81, 21)
            self.ui.connectButton.setStyleSheet("color:rgb(255, 255, 255)")
            self.ui.ipLine.setDisabled(False)
            self.ui.portLine.setDisabled(False)
            self.ui.featureLine.setDisabled(False)
            self.ui.referenceLine.setDisabled(False)
            self.ui.bufferLine.setDisabled(False)
            self.ui.kernelBox.setDisabled(False)
            self.ui.gammaLine.setDisabled(False)
            self.ui.degreeLine.setDisabled(False)
            self.ui.eigLine.setDisabled(False)
            self.ui.wcssLine.setDisabled(False)
            self.mqtt_connected = False
            self.client.terminate()

    @pyqtSlot()
    def drawPred(self):
        self.predfigure.clear()
        axpred = self.predfigure.add_subplot(111)
        axpred.plot(self.pred, 'o-', c='lightskyblue')
        axpred.set_ylim(-0.25, 1.25)
        axpred.set_yticks((0, 1))
        axpred.set_yticklabels(('normal', 'faulty'))
        self.predcanvas.draw()

    normal_drawn = False
    @pyqtSlot()
    def drawX(self):
        if self.normal_drawn == False:
            self.reference_to_draw = self.normal_reference
            self.reference_samples = np.shape(self.reference_to_draw)[1]
            self.Xfigure.clear()
            axX = self.Xfigure.add_subplot(111)
            axX.scatter(self.reference_to_draw[0], self.reference_to_draw[1], s = 25, c = 'lightskyblue')
            self.Xcanvas.draw()
            self.normal_drawn = True
        elif self.normal_drawn == True:
            self.X_to_draw = self.X.T
            self.Xfigure.clear()
            axX = self.Xfigure.add_subplot(111)
            axX.scatter(self.reference_to_draw[0], self.reference_to_draw[1], s = 25, c = 'lightskyblue')
            axX.scatter(self.X_to_draw[0][self.reference_samples:], self.X_to_draw[1][self.reference_samples:], s = 25, c = 'khaki')
            self.Xcanvas.draw()

    draw_SVMPlot = False
    @pyqtSlot()
    def drawSVM(self):
        if self.draw_SVMPlot == True:
            self.X_to_drawPred = self.X[self.reference_samples:]
            self.SVMfigure.clear()
            axSVM = self.SVMfigure.add_subplot(111)
            axSVM.scatter(self.reference_to_draw[0], self.reference_to_draw[1], s = 25, c = 'lightskyblue')
            axSVM.scatter(self.X_to_drawPred[self.pred == 0, 0], self.X_to_drawPred[self.pred == 0, 1], s = 25, c = 'seagreen')
            print("Normal Drawn")
            axSVM.scatter(self.X_to_drawPred[self.pred == 1, 0], self.X_to_drawPred[self.pred == 1, 1], s = 25, c = 'tomato')
            print("Fault Drawn")
            self.SVMcanvas.draw()

    #System Status
    counter = 0
    @pyqtSlot(list)
    def GUI_dispatcher(self, fd_dispatch_result):
        print("reached from GUI dispatcher")
        self.counter = self.counter + 1
        self.ui.samplesLine.setText(str(self.counter))
        if type(fd_dispatch_result[0]) == int: # if unpacked list == int:
            buffer_size = fd_dispatch_result[0]
            self.ui.currentBufferLine.setText(str(buffer_size)) 
        elif type(fd_dispatch_result[0]) == tuple: # if unpacked list == tuple:
            buffer_size = fd_dispatch_result[0]
            if len(fd_dispatch_result[0]) == 2:
                buffer_size, normal_reference = fd_dispatch_result[0]
                self.normal_reference = normal_reference
                self.ui.currentBufferLine.setText(str(buffer_size))
                self.ui.referenceCheck.setChecked(True)
                self.got_normal_reference.emit()
            elif len(fd_dispatch_result[0]) == 4:
                buffer_size, X, wcss_diff, pred = fd_dispatch_result[0]
                self.ui.currentBufferLine.setText(str(buffer_size))
                self.ui.bufferCheck.setChecked(True)
                self.ui.wcssCheck.setChecked(True)
                self.ui.wcssDiffLine.setText(str(wcss_diff))
                self.draw_SVMPlot = True
                self.X = X
                self.pred = pred
                print(np.shape(self.pred))
                self.got_X.emit()
                self.got_pred.emit()
            elif len(fd_dispatch_result[0]) == 3:
                if type(fd_dispatch_result[0][2]) == float:
                    buffer_size, X, wcss_diff = fd_dispatch_result[0]
                    self.ui.currentBufferLine.setText(str(buffer_size)) 
                    self.ui.bufferCheck.setChecked(True)
                    self.ui.wcssDiffLine.setText(str(wcss_diff))
                    self.X = X
                    self.got_X.emit()
                elif type(fd_dispatch_result[0][2]) == list:
                    buffer_size, X, pred = fd_dispatch_result[0]
                    self.ui.currentBufferLine.setText(str(buffer_size)) 
                    self.ui.classifierCheck.setChecked(True)
                    self.X = X
                    self.pred = np.array(pred)
                    print(np.shape(self.pred))
                    self.got_X.emit()
                    self.got_pred.emit()
        fd_dispatch_result = []

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

#if __name__ == "__main__":
#    app = QtWidgets.QApplication(sys.argv)
#    MainWindow = QtWidgets.QMainWindow()
#    ui = Ui_MainWindow()
#    ui.setupUi(MainWindow)
#    MainWindow.show()
#    sys.exit(app.exec_())