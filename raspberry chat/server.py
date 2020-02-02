import time
import socket

# creating a socket object
s = socket.socket(socket.AF_INET,
                  socket.SOCK_STREAM)

# get local Host machine name
host = '192.168.43.104'  
port = 9999

# bind to pot
s.bind((host, port))

# Que up to 5 requests
s.listen(5)

# establish connection
clientSocket, addr = s.accept()
message = input("enter message ::    ")
clientSocket.send(bytes(message,'utf-8'))
clientSocket.close()