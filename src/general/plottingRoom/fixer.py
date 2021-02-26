import os


def fix1(ff):
   with open(ff,"r") as fr:
      print("reading",ff)
      l=fr.readlines()
      ls = (",".join(l) ).split(",")
      print(ls[0:3])
      
      num=""
      numCifre=1
      vecchioValore = 0
      for i,value in enumerate(ls):
         
         if i%2==0 and i != 0:
            # if(i//(2*10*numCifre) > vecchioValore):
            #    vecchioValore = i//(2*10*numCifre)
            #    numCifre+=1
            # num = ls[i][- (1+i//2*(10*numCifre)): ]            
            # print(num)
            # num = ls[i][- (1+i//2*(10*numCifre)): ]            

            ls[i]+="\n"
            print('num: ', num,'numCifre: ', numCifre,'vecchioValore: ', vecchioValore)

      with open("Fix"+ff, "w") as fw:
         fw.write(",".join(ls) )


for ff in os.listdir("./"):
   if(".py" in ff):
      continue
   if ff[0:3]=="Fix":
      
      with open(ff,"r") as fr:
         with open("Fix"+ff, "w") as fw:
            print("reading",ff)
            l=fr.readlines()
            for i,r in enumerate(l) :

               if(i!=0):
                  s = str(i)+r
               else:      
                  s = r
               
               fw.write(s)
               
   

