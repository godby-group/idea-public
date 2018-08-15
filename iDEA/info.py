"""Contains information on version, authors, etc."""

# The short X.Y version.
version = '0.1'
# The full version, including alpha/beta/rc tags.
release = '0.1.0'

def get_sha1():
    """Returns sha1 hash of last commit from git

    Works only, if the code resides inside a git repository.
    Returns None otherwise.
    """
    try:
        import os
        cwd = os.path.dirname(os.path.realpath(__file__))
        import subprocess
        cmd = ['git', 'rev-parse', 'HEAD']
        sha1 = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
        return sha1.strip().decode('ascii')
    except Exception:
        return None

authors = [
    'Rex Godby',
    'Piers Lillystone',
    'James Ramsden',
    'Matt Hodgson',
    'Thomas Durrant',
    'Jacob Chapman',
    'Jack Wetherell',
    'Mike Entwistle',
    'Matthew Smith',
    'Leopold Talirz',
    'Aaron Long',
    'Robbie Oliver',
    'Ewan Richardson',
    'Razak Elmaslmane',
    'Sean Adamson',
]

# sort authors alphabetically
authors.sort(key = lambda n: n.split()[-1])

na = len(authors)
authors_long = ""
authors_short = ""
for i in range(na):
    first, last = authors[i].split()

    if i == 0:
        authors_long += '{}'.format(authors[i])
        authors_short += '{}. {}'.format(first[0].upper(), last)
    elif i < na-1:
        authors_long += ', {}'.format(authors[i])
        authors_short += ', {}. {}'.format(first[0].upper(), last)
    else:
        authors_long += ' and {}'.format(authors[i])
        authors_short += ', {}. {}'.format(first[0].upper(), last)
