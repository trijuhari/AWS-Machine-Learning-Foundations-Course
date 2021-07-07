import random
import string

from numpy.core.defchararray import index




def aaa():
    
    for i in range(0,100): 
        list1 = [random.choice(string.ascii_letters.upper()) for i in range(0,101)]
        list2 = [random.choice(string.ascii_letters.lower()) for i in range(0,101)]
        list11 = [random.choice(string.ascii_letters.upper()) for i in range(0,101)]
        list22 = [random.choice(string.ascii_letters.lower()) for i in range(0,101)]
        print ("Soal nomor {}".format(i+1)) 
        print(list1[random.randint(0,100)]+"\t"+ list11[random.randint(0,100)] +"\t"+ list1[random.randint(0,100)]+"\t"+ list11[random.randint(0,100)]+'\n' +
        list22[random.randint(0,100)]+"\t" + list2[random.randint(0,100)] +"\t" +list22[random.randint(0,100)]+"\t" + list2[random.randint(0,100)] )
        print()
        print("1\t2\t3\t4\t")
        print()
        
 
        
 



def create_soal():
    cx = string.ascii_letters.upper()
    alfabet = [i for i in cx[:26]  ]
    for i in range (1,101):
        huruf_ke = alfabet.index(random.choice(alfabet))
        huruf_tengah =  huruf_ke + random.randint(1,5)
        huruf_akhir =   huruf_tengah + random.randint(1,6)
        if  huruf_akhir <=25  and huruf_akhir < 27  and (huruf_tengah - huruf_ke) != 1 and  (huruf_akhir - huruf_tengah) != 1 :
        
            print()
            print(alfabet[huruf_ke] +"\t"+alfabet[huruf_tengah] + "\t"+ alfabet[huruf_akhir]+ "\n")
            print(alfabet[huruf_ke] +"\t"+"\t"+alfabet[huruf_akhir])
            print()
        else:
            create_soal()
       

create_soal()

def soal3():
    LIST =[10,20]
    
    for i in range (1,101):
        a = random.randint(1,int(random.choice([10,20])))
        b = random.randint(1,int(random.choice([10,20])))
        c= random.randint(1,int(random.choice([10,20])))
        if  a!=b and b!=c :
            print("{}. \n".format(i))
            print(a, "\t A")
            print(b, "\t B")
            print(c, "\t C \n")
        else:
             soal3()
soal3()


 


def create_soallain():
    cx = string.ascii_letters.upper()
    dx = string.ascii_letters.lower
    +                                                                                                                                                               ()
    alfibit = [i for i in dx[:26] ]
    alfabet = [i for i in cx[:26]  ]
    for i in range (1,101):
        huruf_1 = alfabet.index(random.choice(alfabet))
        huruf_2 = alfabet.index(random.choice(alfabet))
        huruf_3 = alfabet.index(random.choice(alfabet))
        huruf_4 = alfabet.index(random.choice(alfabet))
        
        
        if  huruf_1 != huruf_2 and huruf_2 != huruf_3 and  huruf_3 != huruf_4 and huruf_1 != huruf_3 and huruf_1 != huruf_4 and  huruf_2 != huruf_4:
            list =[huruf_1, huruf_2, huruf_3, huruf_4]
            print ("Soal nomor ") 
            print(alfabet[list[random.randint(0,3)]]+"\t"+ alfabet[list[random.randint(0,3)]] +"\t"+ alfabet[list[random.randint(0,3)]]+"\t"+ alfabet[list[random.randint(0,3)]]+'\n' +
        alfibit[list[random.randint(0,3)]]+"\t" + alfibit[list[random.randint(0,3)]] +"\t" +alfibit[list[random.randint(0,3)]]+"\t" + alfibit[list[random.randint(0,3)]] )
            print()
            print("1\t2\t3\t4\t")
            print()
    
        else:
            create_soallain()
    
create_soallain()
