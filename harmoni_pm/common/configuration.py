#
# Copyright (c) 2021 Gonzalo J. Carracedo <BatchDrake@gmail.com>
# 
#
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this 
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from configparser import RawConfigParser

import ast

IMPLIED_SECTION = "main"

class EntryReference:
    def __init__(self, key):
        self.key = key
        
    def __str__(self):
        return self.key
    
    def __repr__(self):
        return str(self)
    
    def resolve(self, config, recursion_limit = 10):
        i = 0
        ref = self
        while i < recursion_limit:
            value = config.get(ref.key)
            if type(value) is EntryReference:
                ref = value
                i += 1
            elif value is None:
                raise LookupError("No such configuration entry `" + ref.key + "'")
            else:
                return value
        raise LookupError("Entry reference recursion limit reached")
    
def _parse_entry(asstr):
    if asstr[0].isalpha():
        return EntryReference(asstr)
    else:
        return ast.literal_eval(asstr)
    
class Entry:
    def __init__(self, parent, datatype, name, default, value = None):
        self.parent = parent
        
        if type(datatype) is not type:
            raise ValueError("datatype must be type (not {0})".format(type(datatype).__name__))
        
        self.type    = datatype
        
        if type(name) is not str:
            raise ValueError("name must be a string (not {0})".format(type(name).__name__))
        
        if type(default) is not datatype:
            raise ValueError(
                "Invalid default value (must be {0}, not {1})".format(
                    self.type.__name__,
                    type(default).__name__))
        
        if value is not None and type(value) is not datatype:
            raise ValueError(
                "Invalud value (must be {0}, not {1})".format(
                    self.type.__name__,
                    type(value).__name__))
        
        
        self.name    = name
        self.default = default
        
        self.value = default if value is None else value
        
    def reset(self):
        self.value   = self.reset
    
    def get(self):
        return self.value
    
    def set(self, value):
        value_type = type(value)
        if value_type is EntryReference:
            value_type = type(value.resolve(self.parent.parent))
        
        expected_type = self.type
        if expected_type is EntryReference:
            expected_type = type(self.value.resolve(self.parent.parent))
            
        if expected_type is not type(None) and value_type is not expected_type:
            raise ValueError(
                "Invalid value (must be {0}, not {1})".format(
                    expected_type.__name__,
                    type(value).__name__))
        self.value = value
        
    def as_str(self):
        return repr(self.value)
    
    def parse(self, asstr):
        # Some quantities may be uncertain with a distribution
        
        self.set(_parse_entry(asstr))
        
class Section:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.entries = {}
        
    def set(self, name, value):
        if not self.have(name):
            # New entry: deduce type and default directly from value
            self.entries[name] = Entry(self, type(value), name, value)
        else:
            # Existing entry. Attempt to set value
            self.entries[name].set(value)
            
    def have(self, name):
        return name in self.entries
    
    def get(self, name):
        if not self.have(name):
            return None
        
        return self.entries[name].get()
    
    def parse(self, name, asstr):
        if not self.have(name):
            self.set(name, _parse_entry(asstr))
        else:
            self.entries[name].parse(asstr)
            
    def as_str(self, name):
        if not self.have(name):
            return None
        
        return self.entries[name].as_str()
    
    def get_entries(self):
        return self.entries.keys()
    
    def as_dict(self):
        dic = {}
        
        for i in self.get_entries():
            dic[i] = self.as_str(i)
            
        return dic
    
    def from_dict(self, dic):
        for i in dic.keys():
            self.parse(i, dic[i])
            
class Configuration:
    def __init__(self):
        self.sections = {}
        
    def have(self, name):
        section, key = self.parse_key(name)
        
        return self.have_section(section) and self.sections[section].have(key)
    
    def copy_from(self, config):
        if type(config) is dict:
            for key in config:
                self[key] = config[key]
        else:
            for i in config.sections.keys():
                section = self.upsert_section(i)
                for j in config.sections[i].get_entries():
                    section.set(j, config.sections[i].get(j))
                    
    def have_section(self, name):
        return name in self.sections
    
    def parse_key(self, name):
        qualified = name.split('.', 1)
        if len(qualified) == 1:
            section = IMPLIED_SECTION
            key     = qualified[0]
        else:
            section = qualified[0]
            key     = qualified[1]
            
        return (section, key)
    
    def upsert_section(self, section):
        if not self.have_section(section):
            self.sections[section] = Section(self, section)
            
        return self.sections[section]
    
    def get(self, name):
        section, key = self.parse_key(name)            
        if not self.have_section(section):
            return None
        
        return self.sections[section].get(key)
    
    def set(self, name, value):
        section, key = self.parse_key(name)            
        s = self.upsert_section(section)
        
        s.set(key, value)
    
    def __getitem__(self, key):
        value = self.get(key)
    
        if type(value) is EntryReference:
            value = value.resolve(self)
        
        return value
    
    def __setitem__(self, key, value):
        self.set(key, value)
        
    def parse(self, name, asstr):
        section, key = self.parse_key(name)            
        s = self.upsert_section(section)
        
        s.parse(key, asstr)
    
    def parse_dict(self, dic):
        for i in dic.keys():
            self.parse(i, dic[i])
            
    def write(self, file):
        cfgfile = RawConfigParser()
        cfgfile.optionxform = lambda option: option
        
        # Convert configuration to dictionary, section by section
        for i in self.sections.keys():
            cfgfile[i] = self.sections[i].as_dict()
            
        with open(file, 'w') as conf:
            cfgfile.write(conf)
    
    def load(self, file):
        cfgfile = RawConfigParser()
        cfgfile.optionxform = lambda option: option

        ok = len(cfgfile.read(file))
        
        if ok == 0:
            raise RuntimeError(
                "Failed to open configuration file `{0}'".format(file))
            
        for s in cfgfile.sections():
            for k in cfgfile[s]:
                self.parse(s + "." + k, cfgfile[s][k])
                