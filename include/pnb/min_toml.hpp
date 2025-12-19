#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pnb
{
namespace toml_min
{

static inline std::string trim(std::string_view s)
{
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))
        ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
        --e;
    return std::string(s.substr(b, e - b));
}

static inline std::string strip_comment(std::string_view s)
{
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i)
    {
        char c = s[i];
        if (c == '"' && (i == 0 || s[i - 1] != '\\'))
            in_str = !in_str;
        if (!in_str && c == '#')
            return std::string(s.substr(0, i));
    }
    return std::string(s);
}

static inline std::string normalize_number_token(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
    {
        if (c == '_')
            continue;
        out.push_back(c);
    }
    return out;
}

struct Table
{
    std::unordered_map<std::string, std::string> kv;
    std::vector<std::string> warnings;

    bool has(std::string_view key) const { return kv.find(std::string(key)) != kv.end(); }

    std::optional<std::string> get_raw(std::string_view key) const
    {
        auto it = kv.find(std::string(key));
        if (it == kv.end())
            return std::nullopt;
        return it->second;
    }

    std::string get_string(std::string_view key, std::string def) const
    {
        auto raw = get_raw(key);
        if (!raw)
            return def;
        std::string v = trim(*raw);
        if (v.size() >= 2 && v.front() == '"' && v.back() == '"')
        {
            std::string out;
            out.reserve(v.size() - 2);
            for (size_t i = 1; i + 1 < v.size(); ++i)
            {
                char c = v[i];
                if (c == '\\' && i + 1 < v.size() - 1)
                {
                    char n = v[i + 1];
                    if (n == 'n')
                        out.push_back('\n');
                    else if (n == 't')
                        out.push_back('\t');
                    else if (n == '"' || n == '\\')
                        out.push_back(n);
                    else
                        out.push_back(n);
                    ++i;
                }
                else
                {
                    out.push_back(c);
                }
            }
            return out;
        }
        return v;
    }

    bool get_bool(std::string_view key, bool def) const
    {
        auto raw = get_raw(key);
        if (!raw)
            return def;
        std::string v = trim(*raw);
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (v == "true")
            return true;
        if (v == "false")
            return false;
        return def;
    }

    int get_int(std::string_view key, int def) const
    {
        auto raw = get_raw(key);
        if (!raw)
            return def;
        std::string v = normalize_number_token(trim(*raw));
        try
        {
            size_t pos = 0;
            int out = std::stoi(v, &pos, 10);
            if (pos != v.size())
                return def;
            return out;
        }
        catch (...)
        {
            return def;
        }
    }

    uint32_t get_u32(std::string_view key, uint32_t def) const
    {
        auto raw = get_raw(key);
        if (!raw)
            return def;
        std::string v = normalize_number_token(trim(*raw));
        try
        {
            size_t pos = 0;
            unsigned long out = std::stoul(v, &pos, 10);
            if (pos != v.size())
                return def;
            if (out > 0xFFFFFFFFu)
                return def;
            return static_cast<uint32_t>(out);
        }
        catch (...)
        {
            return def;
        }
    }

    size_t get_size(std::string_view key, size_t def) const
    {
        auto raw = get_raw(key);
        if (!raw)
            return def;
        std::string v = normalize_number_token(trim(*raw));
        try
        {
            size_t pos = 0;
            unsigned long long out = std::stoull(v, &pos, 10);
            if (pos != v.size())
                return def;
            return static_cast<size_t>(out);
        }
        catch (...)
        {
            return def;
        }
    }

    float get_float(std::string_view key, float def) const
    {
        auto raw = get_raw(key);
        if (!raw)
            return def;
        std::string v = normalize_number_token(trim(*raw));
        try
        {
            size_t pos = 0;
            float out = std::stof(v, &pos);
            if (pos != v.size())
                return def;
            if (!std::isfinite(out))
                return def;
            return out;
        }
        catch (...)
        {
            return def;
        }
    }
};

static inline Table parse_file(const std::string &path)
{
    std::ifstream is(path);
    if (!is)
        throw std::runtime_error("無法開啟config:" + path);

    Table t;
    std::string line;
    std::string prefix;
    size_t line_no = 0;
    while (std::getline(is, line))
    {
        line_no++;
        std::string raw = trim(strip_comment(line));
        if (raw.empty())
            continue;

        if (raw.front() == '[' && raw.back() == ']')
        {
            std::string sec = trim(std::string_view(raw).substr(1, raw.size() - 2));
            if (!sec.empty() && sec.front() == '.' || (!sec.empty() && sec.back() == '.'))
            {
                t.warnings.push_back("config節格式不合法(略過):" + sec);
                prefix.clear();
            }
            else
            {
                prefix = sec;
            }
            continue;
        }

        size_t eq = raw.find('=');
        if (eq == std::string::npos)
        {
            t.warnings.push_back("config行無'='(略過):" + std::to_string(line_no));
            continue;
        }

        std::string key = trim(std::string_view(raw).substr(0, eq));
        std::string val = trim(std::string_view(raw).substr(eq + 1));
        if (key.empty())
        {
            t.warnings.push_back("config鍵為空(略過):" + std::to_string(line_no));
            continue;
        }

        std::string full = prefix.empty() ? key : (prefix + "." + key);
        t.kv[full] = val;
    }

    return t;
}

} // namespace toml_min
} // namespace pnb

