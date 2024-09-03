import os

with open("dummy.txt", "w") as f:
    f.write("Test")
    print("Dummy txt file created!")

# wait for 1 minute
os.system("sleep 60")

print("one minute passed!")