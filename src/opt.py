from optparse import OptionParser

parser = OptionParser()
parser.add_option("--grp:filterbias", default=False,
                  action="store_true", dest="grp_filterbiais",
                  help="do not include 'no' feature in cst nodes")

if __name__ == "__main__":
    (options, args) = parser.parse_args()
    print('test')
    print(args)
    print(options)
