# exploring-replicate
Taking replicate ai for a test run


# Dev steps

1. Clone this repo

2. Install cog

```
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

3. `cd` into repo directory and run: 

```
cog predict -i input="This is so cool!"
```

**Note:** If you're using a lambda cloud gpu instance, you need to run docker (and cog -> docker) commands with `sudo`.


