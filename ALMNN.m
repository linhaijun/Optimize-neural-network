function [DW,DV,Db1,Db2]=ALMNN(SamIn,SamOut,HiddenUnitNum,W,B1,V,b2,sigma,lambda,lr,eta,theta,epsilon,E0,maxepochs,maxk)
%%*******************************************************************************************************
% This is the MATLAB program of a training neural network with Augmented Lagrange Multiplier method, The NN's structure is 1-HiddenUnitNum-1, i.e., the NN has one input and one output, where, HiddenUnitNum is the number of hidden neurons.
% The activation functions in the hidden layer is the sigmoid function and the activation functions in the output layer is the linear function.
% The outputs of this function are the NN's parameters,i.e.,the matrix of NN's weights, and its bias
% The constraint condition is g(A)=dyi/dx>=0, i.e.,g(a)=W.*V>=0
% The parameters are described as following:
% SamIn: input vectors;
% Samout: the target outputs
% HiddenUnitNum: the number of hidden neurons;
% W is the initial matrix of weights connecting the nodes in the input layer with the neurons of the hidden layer;
% B is the initial vector of biases of the neurons of the hidden layer;
% V is the initial vector of weights connecting the neurons of the hidden layer with the ones of the output layer;
% b2 is the initial bias of the neuron of the output layer;
% DW is the matrix of weights connecting the nodes in the input layer with the neurons of the hidden layer after training an NN;
% DV is the vector of weights connecting the neurons of the hidden layer with the ones of the output layer after training an NN;
% Db1 is the vector of biases of the neurons of the hidden layer after training an NN;
% Db2 is the bias of the neuron of the output layer after training an NN;
% sigma is the penalty factor;
% lambda is the multiplier;
% lr is the learning rate;
% eta is the updating factor of the penalty factor; 
% theta is the undating factor of the stopping iteration criterion
% epsilon is the error tolerance of the stopping iteration criterion
% E0 is the target of NN¡¯s mean square error
% Maxepochs is the iteration number for training an NN, i.e., the number of the internal iteration
% maxk is is the iteration number for solving the constrained problem,i.e.,the number of the external iteration

ErrHistory_2=[];
DW=W;
Db1=B1;
DV=V;
Db2=b2;
%maxk=100;
%sigma=sigma_array(sigma_i);
%epsilon=0.00001;
%theta=0.8;
%eta=2.0;
kk=0;
btak=10;       % the initial value of the stopping iteration criterion
btaold=10;     % the second initial value of the stopping iteration criterion
%lambda=1.0*ones(HiddenUnitNum,1);
[InDim,SamNum]=size(SamIn);  %the number of the training samples

while((btak>epsilon)&(kk<maxk)) %
  for train_number=1:maxepochs
    sum_deta_v(1:HiddenUnitNum)=0;   %initializing the NN's parameters
    sum_deta_b2=0;
    sum_deta_w(1:HiddenUnitNum,1:InDim)=0;
    sum_deta_b1(1:HiddenUnitNum)=0;
    HiddenOut(1:HiddenUnitNum)=0;
    for i=1:SamNum  
        sum2=0;     
   %**********************************************************
   %Sovling the NN's output
        for k=1:HiddenUnitNum  
            sum1(1:HiddenUnitNum)=0;
            sum1_E(1:HiddenUnitNum)=0;
            for j=1:InDim        
                sum1(k)=sum1(k)+DW(k,j)*SamIn(j,i);
            end
            sum1_E(k)=sum1(k)+Db1(k);
            HiddenOut(k)=1/(1+exp(-sum1_E(k)));  %the outputs of the hidden layer
            sum2=sum2+DV(k)*HiddenOut(k);
        end
        NetworkOut=sum2+Db2;       %the NN's outputs
        error(i)=SamOut(i)-NetworkOut; %the NN's errors
   %********************************************************************
   %*******************************************************************
   %Sovling the increments of NN's parameters(i.e., W,V,B1,b2) without constraint conditions by using steepest descent method
   %firstly sovling the increments of W and b(2)
   %secondly sovling the increments of V and B1
        deta_v_1=error(i).*HiddenOut*(-1);    % the increment of V, i.e.,the gradients of V
        deta_b2_1=error(i)*(-1);              % the increment of b2, i.e.,the gradients of b2
        sum_deta_v=sum_deta_v+deta_v_1;       % using the batch training method,solving the sums of V and b2, respectively.
        sum_deta_b2=sum_deta_b2+deta_b2_1;
        
        %solving the sums of W and B1, respectively.
        for k=1:HiddenUnitNum
            deta_b1_1(k)=error(i)*(1-HiddenOut(k))*HiddenOut(k)*DV(k)*(-1);%the increment of B1, i.e.,the gradients of B1
            for j=1:InDim   
                deta_w_1(k,j)=deta_b1_1(k)*SamIn(j,i);  % the increment of W, i.e.,the gradients of W
            end
        end  
         sum_deta_w=sum_deta_w+deta_w_1; % using the batch training method,solving the sums of W and B1, respectively.
         sum_deta_b1=sum_deta_b1+deta_b1_1;
      end
      %solving the average increments of W,V,B1 and b2, respectively.
      temp_sum_deta_w_1=sum_deta_w/SamNum;
      temp_sum_deta_b1_1=sum_deta_b1/SamNum;
      temp_sum_deta_v_1=sum_deta_v/SamNum;
      temp_sum_deta_b2_1=sum_deta_b2/SamNum;
  %************************************************************************
  
  %************************************************************************
  %Sovling the increments of NN's parameters(i.e., W,V,B1,b(2)) with Augmented Lagrange Multiplier method and constraint conditions 
      theory_NetworkOut(1:HiddenUnitNum)=0;      %sovling the increments of V and b2 with constraint conditions
      for k=1:HiddenUnitNum 
          temp_sum=0;
          for j=1:InDim        
              temp_sum=temp_sum+DW(k,j); 
          end
          theory_NetworkOut(k)=temp_sum*DV(k);%the constraint condition is W.*V>=0
          theory_error(k)=sigma*theory_NetworkOut(k)-lambda(k);
          if(theory_error(k)<0)           %if the condition is more than zero, updating the NN's parameters
              deta_v_2(k)=theory_error(k)*temp_sum;   %the increment of V with constraint conditions
              deta_b2_2=0;             %the increment of b2 with constraint conditions
              deta_b1_2(k)=0;          %the increment of B1 with constraint conditions
              for j=1:InDim            %the increment of W with constraint conditions
                  deta_w_2(k,j)=theory_error(k)*DV(k); 
              end 
          else                 %if the condition is less than zero, the increments of the NN's parametrs are zero   
              deta_v_2(k)=0; 
              deta_b2_2=0;            
              deta_b1_2(k)=0;
              for j=1:InDim       
                  deta_w_2(k,j)=0; 
              end 
          end
      end
  %************************************************************************
  
  %*********************************************************************
  %Composing the increments of the NN's parameters without constraint conditions and the increments with constraint conditions
  %i.e., obtaining the new increments of the NN's parameters by using the ALMNN algorithm
      temp_sum_deta_w=temp_sum_deta_w_1+deta_w_2;
      temp_sum_deta_b1=temp_sum_deta_b1_1+deta_b1_2;
      temp_sum_deta_v=temp_sum_deta_v_1+deta_v_2;
      temp_sum_deta_b2=temp_sum_deta_b2_1+deta_b2_2;
      % updating the NN's parameters
      DW=DW-lr*temp_sum_deta_w;
      Db1=Db1-lr*temp_sum_deta_b1';
      DV=DV-lr*temp_sum_deta_v;
      Db2=Db2-lr*temp_sum_deta_b2;
      Err=sumsqr(error)/SamNum;
      if Err<=E0    %if the condition of stopping training NN is satisfied (i.e.,the NN's mean square error is less than E0), stop traning the NN. 
          break;
      end
      ErrHistory_2=[ErrHistory_2 Err];
  end
  %*********************************************************************
  
  %*********************************************************************
  %updating some parameters, 
  %updating the penalty factor sigma
  %updating the multiplier lambda;
   btak0=0;
   theory_NetworkOut(1:HiddenUnitNum)=0;
   for k=1:HiddenUnitNum 
       temp_sum=0;
       for j=1:InDim        
           temp_sum=temp_sum+DW(k,j);
       end
       theory_NetworkOut(k)=temp_sum*DV(k);  %the constraint condition is W.*V>=0
       temp=min(theory_NetworkOut(k),lambda(k)/sigma);
       btak0=btak0+temp^2;
   end
   btak=sqrt(btak0)
   if(btak>epsilon)                 %if the stopping iteration criterion is satisfied, updating the penalty factor
       if((kk>2)&(btak>theta*btaold))% updating the penalty factor
           sigma=eta*sigma;
       else 
           sigma=sigma;              %else the updating penalty factor keeps unchanged
       end
       theory_NetworkOut(1:HiddenUnitNum)=0;
       for k=1:HiddenUnitNum          %undating the multiplier
           temp_sum=0;
           for j=1:InDim        
               temp_sum=temp_sum+DW(k,j); 
           end
           theory_NetworkOut(k)=temp_sum*DV(k); %the constraint condition is W.*V>=0
           lambda(k)=max(0,lambda(k)-sigma*theory_NetworkOut(k)); %updating the multiplier.
       end
   end
   kk=kk+1;
   btaold=btak;
   %***********************************************************************
end
DW
Db1
DV
Db2     
number=size(ErrHistory_2,2);
xx1=1:number;
figure;
plot(xx1,ErrHistory_2);      
end

