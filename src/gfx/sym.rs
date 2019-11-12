//! Sym used to identify shader module resources.
use std::borrow::Borrow;
use std::ops::Deref;

pub struct Sym(str);
impl Sym {
    pub fn new<S: AsRef<str> + ?Sized>(literal: &S) -> &Sym {
        unsafe { &*(literal.as_ref() as *const str as *const Sym) }
    }
    pub fn segments<'a>(&'a self) -> Segments<'a> { Segments(self.0.as_ref()) }
}
impl ToString for Sym {
    fn to_string(&self) -> String { self.0.to_string() }
}
impl ToOwned for Sym {
    type Owned = Symbol;
    fn to_owned(&self) -> Symbol { Symbol::from(self.0.to_owned()) }
}
impl AsRef<str> for Sym {
    fn as_ref(&self) -> &str { &self.0 }
}

pub struct Symbol(Box<Sym>);
impl Symbol {
    pub fn segments<'a>(&'a self) -> Segments<'a> { Segments(&(*self.0).0) }
}
impl From<Box<Sym>> for Symbol {
    fn from(boxed: Box<Sym>) -> Symbol {
        Symbol(boxed)
    }
}
impl From<String> for Symbol {
    fn from(literal: String) -> Symbol {
        unsafe { Symbol::from(Box::from_raw(std::boxed::Box::leak(literal.into_boxed_str()) as *mut str as *mut Sym)) }
    }
}
impl ToString for Symbol {
    fn to_string(&self) -> String { self.0.to_string() }
}
impl Borrow<Sym> for Symbol {
    fn borrow(&self) -> &Sym { self }
}
impl Deref for Symbol {
    type Target = Sym;
    fn deref(&self) -> &Sym { &self.0 }
}

pub struct Segments<'a>(&'a str);
impl<'a> Iterator for Segments<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            return None;
        }
        if let Some(pos) = self.0.char_indices()
            .find_map(|(i, c)| if c == '.' { Some(i) } else { None }) {
            let seg = &self.0[..pos];
            self.0 = &self.0[(pos + 1)..];
            Some(seg)
        } else {
            let seg = &self.0[..];
            self.0 = &self.0[self.0.len()..];
            Some(seg)
        }
    }
}
