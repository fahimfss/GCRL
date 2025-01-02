def debug(msg="debug", interrupt=True, postmortem=False, pdb=False):
    if interrupt:
        # import pdb
        import pudb

        spaces = min(len(msg) + 4, 30)
        print(f'{"#"*spaces}')
        # print(f"#{' '*(spaces//2)}debug{' '*(spaces//2)}#")
        print(f"# {msg} #")
        print(f'{"#"*spaces}')
        # print(f'{"#"*(spaces+5)}')
        # not working for now
        # if postmortem:
        #     import traceback
        #     pudb.post_mortem()
        # else:
        if pdb:
            pdb.set_trace()
        else:
            breakpoint()
            
