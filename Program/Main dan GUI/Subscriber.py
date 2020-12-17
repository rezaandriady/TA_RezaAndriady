import paho.mqtt.client as mqtt
from collections import deque
import pandas as pd
import Fault_Detection_Arzy
from PyQt5.QtCore import pyqtSignal, QThread
#import main

class MyMQTTClass(QThread):
    def __init__(self, parent):
        super(MyMQTTClass, self).__init__(parent)
        self.m_client =  mqtt.Client(clean_session=True, protocol=mqtt.MQTTv31)
        self.m_client.on_connect = self.on_connect
        self.m_client.on_message = self.on_message
        # self.m_client.on_disconnect = self.on_disconnect

    def on_connect(self, mqttc, obj, flags, rc):
        print("rc: "+str(rc))

    topic_len = 53
    topics = range(topic_len)
    buffers = [[topic] for topic in topics]
    ############ REFERENCE SIZE HERE
    reference_size = 150
    ############ BUFFER SIZE HERE
    buffer_size = 50
    data_stream = []
    data_pocket = []
    buffer_pocket=[]
    ip = 0
    port = 0
    messageSignal = pyqtSignal(list)
    on_messageSignal = pyqtSignal(list)

    def on_message(self, mqttc, userdata, message):
        self.data_pocket.append(message.payload)
        if (len(self.data_pocket)==len(self.topics)):
            for i in range(len(self.topics)):
                self.buffer_pocket[i].append(self.data_pocket[i])
            self.data_pocket.clear()
            self.data_stream = pd.DataFrame.from_records(self.buffer_pocket)
            self.data_stream = self.data_stream.T
            self.on_messageSignal.emit([self.data_stream.iloc[:,:-1]])
            fd_dispatch_result = Fault_Detection_Arzy.faultDetection.dispatcher(self.data_stream)
            print("Reached Subscribe Results")
            self.messageSignal.emit([fd_dispatch_result]) #send data as a list

    def on_subscribe(self, mqttc, obj, mid, granted_qos):
        print("Subscribed: "+str(mid)+" "+str(granted_qos))

    #receive variables from UI
    def getBuffs(self, buffs):
        self.buffer_size = buffs
        print('subscriber buffer: ', self.buffer_size)

    def getRefs(self, reffs):
        self.reference_size = reffs
        print('subscriber reference: ', self.reference_size)

    def getIp(self, ip):
        self.ip = ip
        print(self.ip)
    
    def getPort(self,port):
        self.port = port
        print(self.port)

    def getTopics(self, topic_len):
        self.topic_len = int(topic_len)
        print('subscriber topics: ',self.topic_len)

    def run(self):
        print([self.topic_len, self.reference_size, self.buffer_size, self.ip, self.port])
        for j in range( len(self.buffers) ):
            buffer = deque(maxlen= self.reference_size)
            self.buffer_pocket.append(buffer)
        try:
            self.m_client.connect(str(self.ip), int(self.port), 60)
            self.m_client.subscribe("#")
            rc = 0
            while rc == 0:
                rc = self.m_client.loop()
            return rc
        except ConnectionRefusedError:
            print("No connection could be made because the target machine actively refused it")