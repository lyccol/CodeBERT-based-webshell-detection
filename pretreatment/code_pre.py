import re



def code_pre(outstring):


    # 清除/** ... **/  注释
    m = re.compile(r'/\*.*?\*/', re.S)
    outtmp = re.sub(m, '', outstring)
    outstring = outtmp


    # 清除//注释
    m = re.compile(r'//.*')
    outtmp = re.sub(m, '', outstring)
    outstring = outtmp


    # 清除#注释
    m = re.compile(r'#.*')
    outtmp = re.sub(m, ' ', outstring)
    outstring = outtmp

    for char in ['\r\n', '\r', '\n']:
        outstring = outstring.replace(char, ' ')

    return outstring