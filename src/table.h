#pragma once 
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

class Table 
{
public:
    Table(std::ostream& s, std::size_t column_width, std::vector<std::string> const& headers)
        : out(s), column_width(column_width), headers(headers)
    {}

    void print_separator() const
    {
        const std::size_t separator_width = (column_width + 3) * headers.size();
        out << std::string(separator_width, '-') << '\n';
    }

    void print_headers() const
    {
        print_separator();

        for(auto const& name : headers)
            out << center_align(name, column_width) << " | ";
        out << '\n';

        print_separator();
    }

    void print_value_row(std::vector<std::string> const& values) const
    {
        if (values.size() != headers.size())
            throw std::invalid_argument("incompatible values Table");
        
        for(auto const& value : values)
            out << center_align(value, column_width) << " | ";
        out << '\n';
    }
private:
    static std::string center_align(std::string const& s, const std::size_t width) 
    {
        std::stringstream ss, spaces;

        const int padding = width - s.size();   

        for(int i = 0; i < padding / 2;++i) spaces << ' ';

        ss << spaces.str() << s << spaces.str();    
        if (padding > 0 && padding % 2!=0)               
            ss << ' ';

        return ss.str();
    }

private:
    std::ostream& out;
    std::size_t column_width;
    std::vector<std::string> headers;
};