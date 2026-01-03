Name:           mimg
Version:        %{version}
Release:        1%{?dist}
Summary:        High-performance command-line image processing tool

License:        MIT
URL:            https://github.com/brian-sinquin/mimg
Source0:        %{name}-%{version}.tar.gz

%description
mimg is a fast image processing tool written in Zig that supports
multiple formats (PNG, JPEG, BMP, QOI) with chainable modifiers
for complex transformations.

Features:
- Fast image processing
- Multiple format support
- Chainable modifiers
- Built-in presets
- Zero external dependencies

%prep
# No prep needed for pre-compiled binary

%build
# No build needed for pre-compiled binary

%install
mkdir -p %{buildroot}/usr/local/bin
install -m 0755 %{_sourcedir}/mimg %{buildroot}/usr/local/bin/mimg

%files
/usr/local/bin/mimg

%changelog
* $(date "+%a %b %d %Y") Automated Build <automated@build.system> - %{version}-1
- Automated build for version %{version}
