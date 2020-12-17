import paho.mqtt.client as mqtt
import time
import csv

def on_log(client,userdata,level,buf):
        print("log: " +buf)

def on_connect(client,userdata,flags,rc):
    if rc == 0:
        print('connected OK')
    else:
        print('Bad Connection Returned code', rc)

def on_disconnect(client,userdata,flags,rc=0):
        print("disconnected result code: " +str(rc))

broker = '192.168.100.42'
client = mqtt.Client('arzyp')

client.on_connect = on_connect
client.on_log = on_log
client.on_disconnect = on_disconnect

print("Connecting to broker", broker)
client.connect(broker)
client.loop_start()

with open('../../Data Pengujian/DatasetRz.csv', newline= '') as file:
    reader = csv.reader(file, delimiter=',')
    data = [row for row in reader]

sensorlist = range(53) # feature size + label
messagelist = data[0:]

#for row in range(0,len(messagelist)):
#    print(row+1)
#    for cell in range(0,len(sensorlist)):
#        topic = str(sensorlist[cell])
#        message = str(messagelist[row][cell])
#        print(topic,message,time.localtime()[4:])
#    time.sleep(10)
    
#publish semua data
for row in range(len(messagelist)):
    for cell in range(len(sensorlist)):
        topic = str(sensorlist[cell])
        message = str(messagelist[row][cell])
        print(message)
        client.publish(topic,message)
    time.sleep(1)

client.loop_stop()
client.disconnect()