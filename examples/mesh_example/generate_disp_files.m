clear all, close all;

% image dimensions
dims = [100 100];

% mapping function
fun = @sinewave2;

% generate Ux and Uy fields
[Y,X]=meshgrid(1:dims(2),1:dims(1));
[fX,fY]=fun(X(:),Y(:));
fX=fX-X(:);
fY=fY-Y(:);
fX=reshape(fX,dims(1),dims(2));
fY=reshape(fY,dims(1),dims(2));

dlmwrite('sinV_p50_s100_Ux.csv', fX);
dlmwrite('sinV_p50_s100_Uy.csv', fY);

%% Display

% figure;
% imagesc(fX);
% colorbar;
% title('displacement Ux');
% axis equal tight;
% 
% figure;
% imagesc(fY);
% colorbar;
% title('displacement Uy');
% axis equal tight;
% 
% stepx=floor(dims(1)/30);
% stepy=floor(dims(2)/30);
% figure;
% quiver(Y(1:stepx:end,1:stepy:end),X(1:stepx:end,1:stepy:end),...
%     fY(1:stepx:end,1:stepy:end),fX(1:stepx:end,1:stepy:end));
% ylabel('x'), xlabel('y');
% title('U field');
% axis ij equal tight;