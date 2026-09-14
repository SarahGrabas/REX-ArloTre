import socket               

sock = socket.socket()         
host = socket.gethostname()
host = ''  #IP adress

port = 4000                 

PATH_IMG ="/Users/sarahgrabas/Desktop/opencv-python/Images/Image0.png"

bytes=1024 #størrelse på billedet

sock.connect((host, port))

file = open(PATH_IMG,'rb')
length = file.read(1024) #læser antal bytes i filen


while (length):
    sock.send(length)
    length = file.read(1024)
    
file.close()

print("Image Send")
sock.shutdown(socket.SHUT_WR)
sock.close() 