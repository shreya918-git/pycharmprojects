def square(num):
    print(num**2)
def main():
    for i in range(1, 11):
        square(i)
if __name__=="__main__": #we can hide the code of modules using this method,because the name would change in another file
    main()