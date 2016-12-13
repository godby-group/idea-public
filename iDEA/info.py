"""Contains information on version, authors, etc."""
version = 'v2.0b'
authors = ['Piers Lillystone', 'James Ramsden', 'Matt Hodgson', 
    'Jacob Chapman', 'Thomas Durrant', 'Jack Wetherell', 'Mike Entwistle',
    'Matthew Smith', 'Leopold Talirz']

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

