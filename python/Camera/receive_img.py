import socket

sock =socket.socket()
host = socket.gethostname()
port=4000 #Skal være samme som på send

sock.bind((host, port))        # Bind to the port
file = open('Image0.png','wb') 