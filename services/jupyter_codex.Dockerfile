FROM jupyter/base-notebook

# install the kernel gateway
RUN pip install jupyter_kernel_gateway

# install additional packages
RUN pip install numpy scipy sympy

# run kernel gateway on container start, not notebook server
EXPOSE 8888
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888"] 