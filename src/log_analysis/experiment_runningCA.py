import os


def parse_results(log_file):
    if os.path.exists(log_file):
        with open(log_file) as file:
            n = False
            for line in file.readlines():
                if n:
                    return line.strip().split("\t")
                if "CL\tTot_q\ttop_lvl_q\tgen_q" in line:
                    n = True
        return ["-"]*12
    return ["NC"]*12

def fetch_list(list_files):
    return [[f[0]]+parse_results(f[1]) for f in list_files]


def plot_table_results(bench, log, prior_name, output_file):
    with open(output_file, "w") as file:
        file.write("\\documentclass[varwidth=\\maxdimen]{standalone}\n\\usepackage{booktabs}\n")
        file.write("\\begin{document}\n\\begin{tabular}{c|cccccccccccc}\n")
        file.write("\\toprule\n")
        file.write(" & CL & Tot\\_q & top\\_lvl\\_q & gen\\_q & fs\\_q & fc\\_q & avg$|$q$|$ & gen\\_time & avg\\_t & max\\_t & tot\\_t & conv\\\\\n")
        file.write("\\midrule\n\\midrule\n")
        list_files = []
        for b in bench:
            prior_name_m = prior_name
            if 'allbut' in prior_name:
                prior_name_m = f"{prior_name}_{b[1]}"
            bn = b[1].replace('_', '\\_')
            pn = prior_name_m.replace('_','\\_')
            list_files.append([f"Baseline {bn}", os.path.join(log,'baseline',f'log_b[{b[1]}].txt')])
            list_files.append([f"PaF {bn} prior {pn}", os.path.join(log,'paf',f'log_withprior_paf_p[{prior_name_m}]_b[{b[1]}].txt')])
            for l in [0.25,0.5,0.75,1]:
                list_files.append([f"PaM {bn} prior {pn} lam {l}", os.path.join(log,'pam',f'log_withprior_pam_{l}_p[{prior_name_m}]_b[{b[1]}].txt')])
        i=0
        for lines in fetch_list(list_files):
            if i % 6 == 0 :
                file.write("\\midrule\n")
            file.write(f"{lines[0]}")
            for v in lines[1:]:
                file.write(f" & {v}")
            file.write("\\\\\n")
            i += 1
        file.write("\\bottomrule\n")

        print("-")

        file.write("\\end{tabular}\n\\end{document}\n")


