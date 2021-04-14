## (C) 2021 Pablo Alvarado
## EL5852 IntroducciÃ³n al Reconocimiento de Patrones
## Tarea 3

## Template file for the whole solution

1;

## Locally weighted regression
##
## p: matrix of size m x 2, with m 2D positions on which
##    the z value needs to be regressed
## X: support data (or training data) with all known 2D positions
## y: support data with the corresponding z values for each position
##
## The number of rows of X must be equal to the length of y
##
## The function must generate the z position for all
function rz=lwr(p,X,z,tau)
  ## This code is for simple linear regression

  ## CHANGE THE FOLLOWING CODE
  ## You have to replace it for proper LWR code
  z=z';
  m = length(z);
  X =[ones(m,1) X]; % Add a column of ones to X
  PP=[ones(rows(p),1) p]; % Add a column of ones to p
  rz = zeros(length(p),length(tau));
  for f=1:length(tau)
    for i = 1:length(p);   
      #% compute weights
      w_ii = exp(-sum((X(:,2:3) - repmat(PP(i,2:3), size(X,1), 1)).^2, 2) / (2*tau(f)^2)); #tomado de https://see.stanford.edu/Course/CS229 materiales ps1_solution
      W = diag(w_ii);
      theta = pinv(X'*W*X)*(X'*W*z);
      rz(i,f)=PP(i,:)*theta;
    end
  end
  #X_s=[ones(rows(X),1),X];
  #m = length(z); % store the number of training examples
  #W=
  #theta=((inv(X_s'*W'*X_s+X_s'*W*X_s))(X_s'*W*z+X_s'*W'*z)); # theta  óptimo
  #PP=[ones(rows(p),1) p];
  #rz=PP*theta; # Es la hipótesis 
  
endfunction

## Use for the experiments just 0,5% of the total available data.
[X,z] = gendata(0.005);

## And the reference data to be used for comparison
[RX,rz] = gendata(1,0,0);

## Extract from RX the used coordinate range:
minx=min(RX(:,1))
maxx=max(RX(:,1))

miny=min(RX(:,2))
maxy=max(RX(:,2))

partition=75;
[xx,yy]=meshgrid(round(linspace(minx,maxx,partition)),
                 round(linspace(miny,maxy,partition)));

## The grid
NX = [xx(:) yy(:)];

printf("Regression started...");
fflush(stdout);
tic();

## Locally weighed regression on the data
## This will take a LONG time once finished
tau = [1 10 25 50 100 200];
nz = lwr(NX,X,z,tau);
printf("done.\n");
toc()
fflush(stdout);

printf("Preparing data to apply the loss function...");
fflush(stdout);
tic();

# Se preparan los datos para evaluar la función de pérdida
rz_new=reshape(rz,1638,2065);
ind=sub2ind(size(rz_new),yy,xx);
rz_ind=rz(ind(:));

printf("done.\n");

# Evaluating the loss function
printf("Evaluating the loss function...");
res = zeros(size(nz,2),1);
for fg=1:size(nz,2)
  res(fg,1)=(1/rows(xx(:)))*norm(nz(:,fg)-rz_ind,'fro')^2; # Loss function MSE
end

printf("done.\n");
for hh=1:size(nz,2)
  printf(strcat('Loss with tau=',num2str(tau(1,hh)),': ',num2str(res(hh,1)),'.\n'));
end
toc()
fflush(stdout);

figure(1,"name","Sensed data");
plot3(X(:,1),X(:,2),z',".");
xlabel("x")
ylabel("y")
zlabel("z")

#plot3(NX(:,1),NX(:,2),nz,"r.");
#surf(xx,yy,reshape(nz(:,2),size(xx)));
#xlabel("x")
#ylabel("y")
#zlabel("z")

figure(2,"name","Regressed data");
hold off;
#plot3(X(:,1),X(:,2),y',"b.");
#hold on;
plot3(NX(:,1),NX(:,2),nz(:,2),"r.");
#surf(xx,yy,reshape(ny,size(xx)));
xlabel("x")
ylabel("y")
zlabel("z")

figure(3,"name","Data resulting from linear regression");
hold off;
#plot3(NX(:,1),NX(:,2),nz,"r.");
surf(xx,yy,reshape(nz(:,2),size(xx)));
xlabel("")
ylabel("y")
zlabel("z")

figure(4,"name","Data resulting from linear regression");
hold off;
plot(tau',res,"b-");
xlabel("tau")
ylabel("Loss")
zlabel("z")