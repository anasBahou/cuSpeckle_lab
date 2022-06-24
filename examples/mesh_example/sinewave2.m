function [x,y]=sinewave2(x,y)
A=5;
f=1/50;
x=x+A*sin(2*pi*f*x);

