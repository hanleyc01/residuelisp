"""Module containing both the encoding functions as well as the interpretation
functions. 

## Encoding

The main way to interact with the `.encoding` module is to call `.encoding.encode` 
the result of `syntax.parser.parse`. These are `syntax.parser.Intr`, or, 
intermediate representations. To call encode, we need to specify an 
`.encoding.EncodingEnvironment`.

The encoding scheme for integers used is given to the environment using
the `.encoding.IntegerEncodingScheme` enumeration.

```
from syntax.parser import lex, parse
from .encoding import EncodingEnvironment, IntegerEncodingScheme, encode
from vsa.hrr import HRR

src = "(cons nil nil)"
intr = parse(lex(src))
env = EncodingEnvironment(HRR, dim=100)
encode(intr, env)
```

## Interpretation

The module for interpretation is `.interpreter`.
"""
from .encoding import *
from .interpreter import *
