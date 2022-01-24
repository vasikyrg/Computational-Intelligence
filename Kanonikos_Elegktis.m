clc
clear all

%Vasikes Parametroi
Kp=1.262;
c=-0.29;
Ka=1;
K=25*Ka*Kp;
Ki=Kp*(-c);

%Synartiseis Metaforas
Gc=zpk(c, 0, Kp);
Gp=zpk([],[-0.1 -10],25);

%Gewmwtrikos topos rizwn
anoixtou_vrohou=Gc*Gp;
figure
rlocus(anoixtou_vrohou);

%Step
kleistou_vrohou=feedback(anoixtou_vrohou,1 ,-1);
figure 
step(kleistou_vrohou)