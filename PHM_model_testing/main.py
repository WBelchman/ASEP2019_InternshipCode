import Random_Forest
import Isolation_Forest
import OCSVM
import SVM
import RFE
import Lin_reg
import ARIMA

#Console UI
def main():

    print("\nPHM_model_testing")
    
    while True:
        user = input("1. Random Forest\n2. Isolation Forest\n3. OCSVM\n4. Classifier SVM\n5. RFE\n6. Linear Regression\n7. Quit\n")
        
        if user == '1':
            Random_Forest.run()
        elif user == '2':
            Isolation_Forest.run()
        elif user == '3':
            OCSVM.run()
        elif user == '4':
            SVM.run()
        elif user == '5':
            RFE.run()
        elif user == '6':
            Lin_reg.run()
        elif user == '7':
            break

        print("\n----------Main-----------")

main()