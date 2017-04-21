"""Contains information on version, authors, etc."""

# The short X.Y version.
version = '2.1'
# The full version, including alpha/beta/rc tags.
release = '2.1.0'


authors = [
    'Piers Lillystone', 
    'James Ramsden', 
    'Matt Hodgson', 
    'Thomas Durrant',
    'Jacob Chapman', 
    'Jack Wetherell', 
    'Mike Entwistle', 
    'Matthew Smith',
    'Aaron Long',
    'Leopold Talirz'
]

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

