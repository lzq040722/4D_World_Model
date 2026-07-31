ssh -N -o ExitOnForwardFailure=yes -L 17778:127.0.0.1:7778 -p 40031 root@10.130.129.33

python -m http.server 8000 --bind 127.0.0.1